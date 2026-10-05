# RAG híbrido con Pinecone sobre el reglamento del Edificio Riesco 6540

Este repo es mi Pre-entrega 4. Armé un módulo de recuperación que sube documentos a un índice **Pinecone Serverless**, los busca combinando **BM25** (búsqueda por palabras) con **búsqueda vectorial** (búsqueda por significado) a través de un `EnsembleRetriever`, y mide qué tan bien funciona con **Recall@5** y **Precision@5** sobre un *golden set* de 5 preguntas.

Como dataset no usé la documentación de una librería, sino el reglamento de copropiedad de un edificio: 4 documentos, 21 secciones y 22 chunks una vez procesados. Lo elegí a propósito. Es un texto normativo lleno de referencias exactas como "Artículo Octavo, letra x)", "UTM" o "departamentos 101 y 104", y en ese tipo de texto la búsqueda semántica sola se queda corta. Era un buen terreno para ver si el enfoque híbrido realmente aporta.

## Resultados en corto

Con el sistema híbrido (pesos 0.5/0.5, k = 5):

| Métrica | Valor |
|---|---|
| Recall@5 | **100 %** (las 5 preguntas) |
| Precision@5 | **52 %** |
| Sección correcta dentro del top-5 | 100 % |
| Tests | 62 pasan con credenciales; sin credenciales pasan 58 y los 4 de integración se saltan |

El detalle completo está en [`resultados/evaluacion.txt`](resultados/evaluacion.txt), y más abajo explico [cómo leer estos números](#qué-me-dicen-los-números), porque el 100 % de recall no significa lo que parece.

---

## Cómo está organizado

El flujo es este:

```
data/*.txt ──► ingest.py ──► 22 chunks ──┬──► Pinecone Serverless (embeddings de 1536, en un namespace)
               corta por sección         │            │
               y luego por tokens        │            ▼  retriever vectorial
                                         │
                                         └──► BM25 en memoria (los mismos chunks)
                                                      │  retriever léxico
                                                      ▼
                       RAGSystem: EnsembleRetriever (RRF, pesos 0.5/0.5, id_key="chunk_id")
                                                      │
                                     ┌────────────────┼────────────────┐
                                     ▼                ▼                ▼
                               evaluate.py     consultar.py      rag_system.py
                               (métricas)      (consola)         (demostración)
```

Y los archivos:

- [`config.py`](config.py): lee el `.env`, valida las credenciales y concentra las constantes (modelo, dimensión, tamaño de chunk, namespace).
- [`setup_index.py`](setup_index.py): crea el índice serverless si no existe. Si ya existe, revisa que su dimensión coincida con la del modelo.
- [`data.py`](data.py): genera los 4 archivos de `data/`. Ya vienen incluidos, así que no hace falta ejecutarlo.
- [`ingest.py`](ingest.py): parsea los documentos, arma los chunks y los sube a Pinecone.
- [`retrievers.py`](retrievers.py): el preprocesado del texto y la construcción de los dos retrievers.
- [`rag_system.py`](rag_system.py): la clase `RAGSystem`, que los combina con `EnsembleRetriever`.
- [`evaluate.py`](evaluate.py): calcula las métricas y compara distintas configuraciones.
- [`consultar.py`](consultar.py): una consola para hacerle preguntas al sistema.
- [`golden_set.json`](golden_set.json): las 5 preguntas de prueba.
- [`tests/`](tests/): 62 tests con pytest.
- [`resultados/`](resultados/): las salidas reales de cada script.

---

## Cómo replicarlo

**1. Entorno.** Necesitas **Python 3.13**. Lo aprendí a la mala: `langchain-pinecone` todavía no es compatible con 3.14, y con esa versión pip termina instalando una versión de 2023 que no funciona.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip check
```

**2. Credenciales.** Copia la plantilla y completa tus datos:

```bash
cp .env.example .env
```

Necesitas `OPENAI_API_KEY` (para los embeddings), `PINECONE_API_KEY` (la sacas de [app.pinecone.io](https://app.pinecone.io)) y `PINECONE_INDEX_NAME`. Ojo con el nombre del índice: Pinecone solo acepta minúsculas, números y guiones. Mi primer intento fue `edificio_riesco` y falló por el guion bajo, así que agregué esa validación en `config.py`.

**3. Crear el índice y cargar los datos.**

```bash
python setup_index.py   # índice serverless en aws/us-east-1, dimensión 1536, métrica coseno
python ingest.py        # sube los 22 chunks al namespace "consultas-convivencia"
```

Puedes ejecutar `ingest.py` las veces que quieras: sobrescribe los mismos vectores, no los duplica.

**4. Probarlo.**

```bash
python rag_system.py    # demostración con 4 consultas
python consultar.py     # modo interactivo (escribe 'salir' para terminar)
python evaluate.py      # métricas y comparación → resultados/evaluacion.txt
pytest -v               # los 62 tests
```

**Sin credenciales.** Si no tienes keys a mano, igual puedes ejecutar `pytest`: los 58 tests offline pasan y los 4 que necesitan Pinecone se saltan solos. Además dejé todas las salidas reales guardadas en [`resultados/`](resultados/), y los docstrings de `rag_system.py` y `consultar.py` traen ejemplos de lo que imprimen.

---

## Las decisiones que tomé (y por qué)

### Cortar por sección antes que por tamaño

Los documentos tienen secciones con título propio, por ejemplo "Procedimiento Sancionatorio y Régimen de Multas". Si los cortaba a ciegas cada 600 tokens, iba a terminar con chunks que mezclan dos temas, como mascotas y multas, y eso ensucia los embeddings.

Por eso `parsear_documento()` primero separa cada archivo por sus secciones (lo que va antes del primer título queda como "Introducción") y recién después aplica un `RecursiveCharacterTextSplitter` sobre cada sección, con `chunk_size=600` y `chunk_overlap=100` medidos en tokens. En la práctica, casi todas las secciones caben en un solo chunk. El splitter solo tuvo que partir una, la de la piscina, que es la más larga.

Algunos chunks, sobre todo las introducciones, quedaron más chicos que los ~500 tokens que sugiere la consigna. Lo dejé así a conciencia: prefiero que cada chunk sea un tema completo antes que forzar un tamaño parejo.

Un detalle que descubrí revisando la ingesta: `from_tiktoken_encoder` usa por defecto el tokenizador `gpt2`, que con texto en español cuenta un 23 % más de tokens que `cl100k_base`, el que usa el modelo de embeddings. Sin fijar `encoding_name`, mis chunks "de 600 tokens" en realidad eran de unos 490.

### Qué guardo en cada vector

Cada vector en Pinecone lleva `doc_id`, `source`, `titulo`, `seccion`, `chunk_index` y `chunk_id`. Además, `PineconeVectorStore` guarda el texto del chunk en `metadata["text"]`, así que cuando recupero un resultado ya tengo el contenido sin consultar ninguna otra base de datos. El `doc_id` es el que comparo con el documento esperado en la evaluación, y `seccion` me permitió medir algo más fino que el documento completo.

### IDs fijos para no duplicar

En la primera versión de la ingesta no le pasaba IDs a Pinecone, así que cada vector iba a recibir uno aleatorio. Al revisarla antes de ejecutarla me di cuenta del problema: cada nueva ingesta habría duplicado todos los vectores, y eso arruina la Precision@5 porque el top-5 se llena de copias del mismo chunk. Ahora cada chunk tiene un ID estable con la forma `convivencia_privada-003`, y volver a ingestar sobrescribe. Lo comprobé: después de dos ejecuciones, el namespace sigue teniendo 22 vectores, no 44.

### Embeddings y namespace

Uso `text-embedding-3-small` con `dimensions=1536`, y esa dimensión sale de la misma constante con la que se crea el índice. Así no hay forma de que el modelo y el índice queden desalineados. Los datos viven en el namespace `consultas-convivencia` para no mezclarse con otros usos del índice.

### El preprocesado que hizo funcionar a BM25

Esta fue la parte más reveladora del proyecto. La primera versión del sistema híbrido daba resultados raros, y al investigar descubrí que BM25 prácticamente no estaba aportando nada.

`BM25Retriever` separa el texto en palabras con un simple `split()`. Con este corpus, eso significa que `"(UTM)."` y `"UTM?"` cuentan como palabras distintas de `"utm"`, y que `"Multas"` no coincide con `"multa"`. En la pregunta "¿Puedo tener mascota?", **ningún** chunk coincidía con nada y todos puntuaban cero. Aun así, BM25 devolvía 5 chunks cualesquiera (todos empatados en cero), y el ensemble los trataba como resultados legítimos, empujando hacia abajo los aciertos de la búsqueda vectorial.

Para corregirlo escribí `preprocesar()`: pasa a minúsculas, quita tildes, extrae solo las palabras, descarta stopwords y quita la "s" final de los plurales. La diferencia, mirando solo BM25 (posición del chunk correcto; "–" significa que ni siquiera aparece en el top-5):

| Consulta | Sin preprocesar | Con `preprocesar()` |
|---|---|---|
| ¿Puedo tener mascota? | – (todo puntuaba 0) | 1.º |
| ¿Qué dice el reglamento sobre las UTM? | – | 1.º |
| Departamentos 101 y 104 | 2.º | 1.º |
| ¿Puedo hacer asados? | 4.º | 2.º |

### Por qué los pesos son 0.5 / 0.5

`RAGSystem` combina los dos retrievers con *Reciprocal Rank Fusion*: cada chunk suma `peso / (posición + 60)` por cada lista en la que aparece. Le puse `id_key="chunk_id"` para que, si el mismo chunk llega desde BM25 y desde Pinecone, se reconozca como uno solo y sume ambos aportes. Por defecto, LangChain compara los documentos por su texto.

Quería elegir los pesos con datos, así que probé varias combinaciones (ver la tabla más abajo). Me llamó la atención que 0.3/0.7 diera prácticamente lo mismo que usar solo el retriever vectorial, y 0.7/0.3 lo mismo que usar solo BM25. La explicación está en la fórmula: con `c = 60` y listas de solo 5 resultados, la posición casi no pesa (entre el 1.º y el 5.º hay un 6 % de diferencia), mientras que los pesos sí (0.7 contra 0.3 es más del doble). El último resultado del retriever "pesado" le gana al primero del "liviano".

En la práctica, con pesos desiguales el top-5 queda formado por los chunks del retriever dominante, y el otro solo puede reordenar los que ambos tienen en común. Mi primera explicación fue que el híbrido "copiaba" al retriever dominante, y fue un test el que me corrigió: el conjunto de chunks sí se copia, pero el orden puede cambiar. Los tests `test_pesos_desiguales_*` dejan documentada la regla correcta. La conclusión es que solo 0.5/0.5 combina de verdad los dos enfoques, y por eso me quedé con esos pesos.

### Pensado para poder testearlo

La entrega anterior me dejó dos sugerencias: documentar salidas de ejemplo para quien no tenga credenciales y agregar tests con pytest. Para que eso fuera posible tuve que ordenar el código:

- Las credenciales se validan recién cuando hace falta conectarse (al crear el índice, al subir los datos, al crear el retriever vectorial). Todo lo demás (parsear, armar los chunks, BM25, las métricas) funciona sin keys.
- `RAGSystem` puede recibir los retrievers ya construidos. Eso me permite probar la fusión con un retriever vectorial falso, sin internet, y comparar configuraciones en `evaluate.py` sin reconstruir todo cada vez.
- Las métricas son funciones simples que no dependen de Pinecone.

---

## Evaluación

### Las preguntas

| # | Pregunta | Documento esperado | Sección esperada | Tipo |
|---|---|---|---|---|
| 1 | ¿A cuántos centímetros del suelo debo guardar las cosas en la bodega? | `convivencia_privada` | Destino Exclusivo Habitacional y Prohibición Comercial | léxica |
| 2 | Si me atraso en pagar los gastos comunes, ¿me pueden cortar la luz? | `obras_mudanzas` | Gastos Comunes, Prorrateo Centralizado y Corte de Suministro | semántica |
| 3 | ¿Puedo llevar amigos que no viven en el edificio a la piscina? | `espacios_comunes` | Régimen Integral de la Piscina del Condominio | semántica |
| 4 | ¿Se pueden hacer mudanzas los domingos? | `obras_mudanzas` | Protocolo de Mudanzas y Carga en Ascensores | léxica |
| 5 | ¿Está permitido guardar materiales inflamables en la terraza? | `asados` | Prevención de Incendios y Materiales Inflamables | léxica |

Las escribí con algunos cuidados:
- Cada una tiene una sola respuesta correcta. Revisé que el dato clave aparezca en un único chunk.
- Mezclé preguntas que usan las mismas palabras del reglamento con otras que lo parafrasean. Por ejemplo, el reglamento nunca dice "cortar la luz", dice "suspender el suministro eléctrico".
- Cubren los 4 documentos.
- Varias tienen palabras trampa: "bodega", "domingo" y "terraza" también aparecen en otras secciones.
- Las escribí **antes** de ver los resultados y no las toqué después. Si una pregunta fallaba, la idea era reportarlo, no reescribirla hasta que pasara.

### Cómo mido

- **Recall@5:** vale 1 si algún chunk del documento correcto aparece entre los 5 primeros.
- **Precision@5:** cuántos de esos 5 son del documento correcto, dividido por 5.
- Como extra, mido si la **sección** exacta aparece en el top-5.

### Comparación de configuraciones

```
Configuración         Recall@5   Prec@5  Sección     #1    #2    #3    #4    #5   ← Precision@5 por pregunta
BM25 solo                 100%      56%      80%    60%   20%   60%   60%   80%
Vectorial solo            100%      52%     100%    20%   40%   60%   60%   80%
Híbrido 0.3 / 0.7         100%      52%     100%    20%   40%   60%   60%   80%
Híbrido 0.5 / 0.5         100%      52%     100%    40%   40%   60%   40%   80%
Híbrido 0.7 / 0.3         100%      56%      80%    60%   20%   60%   60%   80%
```

### Qué me dicen los números

El 100 % de Recall@5 hay que tomarlo con pinzas. Con solo 4 documentos y unos 5 chunks por documento, el correcto casi siempre se cuela en el top-5, así que esta métrica no distingue entre configuraciones. Lo anticipé cuando armé el dataset, y se confirmó.

La que sí aporta información es Precision@5. Mirando pregunta por pregunta, cada retriever falla en un lugar distinto: BM25 se equivoca en la #2, la de "cortar la luz" (20 %), porque esas palabras no están en el reglamento. El vectorial se equivoca en la #1 (20 %), donde la respuesta depende de un dato exacto: "diez centímetros".

El híbrido 0.5/0.5 no gana en el promedio (52 % contra 56 % de BM25), pero nunca baja del 40 % en ninguna pregunta y siempre encuentra la sección correcta. Para mí ese es el argumento real a favor del enfoque híbrido: no mejora el mejor caso, pero protege contra el peor.

También hay que ser honesto con el tamaño de la muestra: con 5 preguntas, la diferencia entre 52 % y 56 % es un solo chunk en una sola pregunta. No alcanza para sacar conclusiones fuertes. Si siguiera con este proyecto, lo primero sería ampliar el golden set antes de afinar más los pesos.

---

## Tests

```bash
pytest -v                        # los 62 (los de integración se saltan si no hay credenciales)
pytest -m "not integration"      # solo los 58 offline
```

Los organicé por módulo:

- `test_config.py` (13): validación del nombre del índice, variables faltantes y configuración de los embeddings.
- `test_ingesta.py` (15): detección de títulos, separación en secciones, los 22 chunks, IDs únicos y estables, tamaño máximo en tokens.
- `test_retrievers.py` (13): quitar tildes, preprocesar, que BM25 encuentre términos exactos y que realmente use mi función de preprocesado.
- `test_rag_system.py` (5): la fusión con un retriever vectorial falso, sin duplicados y con el efecto de los pesos.
- `test_evaluate.py` (8): métricas calculadas a mano, validez del golden set y la evaluación con un sistema de prueba.
- `test_consultar.py` (4): una sesión interactiva simulada y el mensaje de error cuando faltan credenciales.
- `test_integracion.py` (4): contra Pinecone real: dimensión, cantidad de vectores, resultados estables y recall del golden set.

Varios de estos tests nacieron de errores reales que cometí y que no lanzaban ninguna excepción. El más traicionero: escribí `preprocess_fn` en lugar de `preprocess_func`, y BM25 simplemente ignoró el parámetro sin avisar. Otro: una versión de `preprocesar` devolvía un string en vez de una lista, y BM25 terminaba indexando letra por letra. Ahora hay un test para cada uno.

---

## Lo que todavía no resuelve

- **BM25 se confunde con verbos comunes.** Con solo 22 chunks, cualquier palabra que aparece una sola vez se vuelve "importante". En "¿Puedo hacer asados?", BM25 le da a "hacer" el mismo peso que a "asado", y la sección de *Humo y Ruido* sube al primer lugar (la de asados queda segunda). Podría haber agregado "hacer" a las stopwords, pero eso sería acomodar el sistema a mis preguntas de prueba, así que preferí dejarlo documentado.
- **El stemming es básico.** Quitar la "s" final une "multas" con "multa", pero el español tiene muchas otras flexiones. Un stemmer de verdad, como `SnowballStemmer("spanish")`, lo haría mejor a cambio de sumar una dependencia.
- **Hay preguntas ambiguas.** "Departamentos 101 y 104" aparece en dos secciones distintas y las dos son respuestas válidas. Por eso evité ese tipo de preguntas en el golden set.
- **Vectores huérfanos.** Si edito un documento y genera menos chunks que antes, los IDs que sobran se quedan en Pinecone. Lo correcto sería vaciar el namespace antes de volver a ingestar.
- **`langchain-community` está siendo discontinuado** y muestra un aviso al importar `BM25Retriever`. Con las versiones fijadas en `requirements.txt` funciona sin problemas.
- **Versiones fijadas a propósito.** `langchain-pinecone==0.2.13` exige Python menor a 3.14 y `pinecone` menor a 8, por eso uso `pinecone==7.3.0` y no la última.

---

## Salidas de ejemplo

Todas son ejecuciones reales y quedaron guardadas para que se puedan revisar sin credenciales:

- [`resultados/ingesta.txt`](resultados/ingesta.txt): `python ingest.py`
- [`resultados/demo_rag.txt`](resultados/demo_rag.txt): `python rag_system.py` (también está en su docstring)
- [`resultados/sesion_interactiva.txt`](resultados/sesion_interactiva.txt): `python consultar.py` (un extracto está en su docstring)
- [`resultados/evaluacion.txt`](resultados/evaluacion.txt): `python evaluate.py`
- [`resultados/pytest.txt`](resultados/pytest.txt): `pytest -v`

Un ejemplo de la consola interactiva, que muestra bien lo que buscaba con el sistema híbrido:

```
🧑 Tú: ¿Me pueden multar si fumo en el pasillo?

📎 Top 5 fragmentos recuperados:

  1. [convivencia_privada | Normas de Convivencia, Emisiones de Humo y Ruido]  (convivencia_privada-003)
     La tranquilidad y el descanso de los vecinos configuran una obligación esencial contemplada en ...

  2. [convivencia_privada | Procedimiento Sancionatorio y Régimen de Multas]  (convivencia_privada-004)
     Cualquier transgresión a las obligaciones descritas en el Artículo Octavo o en los artículos ...
```

Los dos primeros resultados se complementan: el primero dice que está prohibido fumar en las áreas comunes y el segundo explica cuánto es la multa.
