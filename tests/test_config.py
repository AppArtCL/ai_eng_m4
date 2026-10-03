import pytest
import config


# Credenciales completas e independientes del .env local (los tests deben pasar también sin .env).
@pytest.fixture
def credenciales_ficticias(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-ficticia")
    monkeypatch.setattr(config, "PINECONE_API_KEY", "pcsk-ficticia")
    monkeypatch.setattr(config, "PINECONE_INDEX_NAME", "indice-ficticio")


@pytest.mark.parametrize("nombre", ["edificio-riesco", "indice1", "a", "a" * 45])
def test_nombre_de_indice_valido(credenciales_ficticias, monkeypatch, nombre):
    monkeypatch.setattr(config, "PINECONE_INDEX_NAME", nombre)
    config.validar_credenciales()  # no debe lanzar


@pytest.mark.parametrize("nombre", ["edificio_riesco", "Edificio-Riesco", "edificio riesco", "a" * 46])
def test_nombre_de_indice_invalido(credenciales_ficticias, monkeypatch, nombre):
    monkeypatch.setattr(config, "PINECONE_INDEX_NAME", nombre)
    with pytest.raises(ValueError, match="inválido"):
        config.validar_credenciales()


@pytest.mark.parametrize("variable", ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"])
def test_falta_una_variable(credenciales_ficticias, monkeypatch, variable):
    monkeypatch.setattr(config, variable, None)
    with pytest.raises(ValueError, match=variable):
        config.validar_credenciales()


def test_get_embeddings_sin_credenciales_falla(sin_credenciales):
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        config.get_embeddings()


def test_get_embeddings_usa_modelo_y_dimension_de_config(credenciales_ficticias):
    # Crear el objeto no llama a la API.
    embeddings = config.get_embeddings()
    assert embeddings.model == config.EMBEDDING_MODEL
    assert embeddings.dimensions == config.DIM
