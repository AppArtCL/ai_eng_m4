"""Tests offline del modo interactivo: se reemplaza RAGSystem y la entrada de teclado."""
import io
import pytest
import consultar


def test_resumir_quita_encabezado_y_compacta_espacios():
    contenido = "Título de la sección\nPrimera\tlínea.\n\n   Segunda línea."
    assert consultar.resumir(contenido) == "Primera línea. Segunda línea."


def test_resumir_trunca_textos_largos():
    resumen = consultar.resumir("Encabezado\n" + "x" * 500, largo=50)
    assert resumen == "x" * 50 + "..."


def test_sesion_completa(monkeypatch, capsys, chunks_por_id):
    class RAGFalso:
        def retrieve(self, query):
            return [chunks_por_id["convivencia_privada-004"]]

    monkeypatch.setattr(consultar, "RAGSystem", RAGFalso)
    monkeypatch.setattr("sys.stdin", io.StringIO("¿Cuánto es la multa?\n\nsalir\n"))

    consultar.main()
    salida = capsys.readouterr().out

    assert "¿Cuánto es la multa?" in salida                # entrada no interactiva: se repite la pregunta
    assert "convivencia_privada-004" in salida
    assert "Procedimiento Sancionatorio y Régimen de Multas" in salida
    assert salida.count("📎 Top") == 1                      # el Enter vacío no dispara una consulta
    assert "Listo, terminamos la sesión" in salida


def test_sin_credenciales_termina_con_mensaje_claro(monkeypatch, capsys):
    def falla():
        raise ValueError("OPENAI_API_KEY no puede ser None o vacío.")

    monkeypatch.setattr(consultar, "RAGSystem", falla)
    with pytest.raises(SystemExit) as salida:
        consultar.main()

    assert salida.value.code == 1
    assert ".env.example" in capsys.readouterr().out
