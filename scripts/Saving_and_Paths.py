import os
import json
import torch
import pandas as pd


def get_experiment_path(base_dir="../experiments", parking_id=None, modelo=None, tipo=None):
    """
    Construye la ruta donde se guardarán los resultados del experimento.

    Args:
        base_dir (str): Carpeta base donde se guardan los experimentos.
        parking_id (int or str): ID del parking.
        modelo (str): Nombre del modelo ("lstm", "gru", "rnn").
        tipo (str): "vanilla" o "optuna".

    Returns:
        str: Ruta completa del experimento.
        EJEMPLO: ../experiments/6/lstm/optuna
    """
    assert parking_id is not None and modelo is not None and tipo is not None, "Faltan argumentos para definir la ruta"

    # Estandarizar nombres
    parking_folder = f"parking_{parking_id}"
    modelo = modelo.lower()
    tipo = tipo.lower()

    return os.path.join(base_dir, parking_folder, modelo, tipo)

def save_experiment_results(
    base_dir,
    model,
    history,
    final_metrics,
    save_model=True
):
    """
    Guarda los resultados del experimento: modelo, historial y métricas finales en su carpeta correspondiente

    Args:
        base_dir (str): Ruta donde guardar los archivos.
        model (nn.Module): Modelo entrenado.
        history (list): Lista de dicts con métricas por época.
        final_metrics (dict): Métricas finales de validación.
        save_model (bool): Si guardar o no el modelo (.pt).
    """
    os.makedirs(base_dir, exist_ok=True)

    # Guardar modelo
    if save_model:
        model_path = os.path.join(base_dir, "model.pt")
        torch.save(model.state_dict(), model_path)

    # Guardar historial
    history_path = os.path.join(base_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Guardar métricas finales
    metrics_path = os.path.join(base_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)

def log_experiment_summary(
    csv_path,
    parking_id,
    modelo,
    tipo,
    final_metrics,
    best_params=None
):
    """
    Añade un experimento al archivo CSV resumen global.

    Args:
        csv_path (str): Ruta del archivo CSV de resumen.
        parking_id (int): ID del parking.
        modelo (str): Nombre del modelo ("rnn", "lstm", "gru", etc.).
        tipo (str): "vanilla" o "optuna".
        final_metrics (dict): Métricas finales del experimento.
        best_params (dict, optional): Parámetros optimizados (si aplica).
    """

    # Cargar el CSV si existe, sino crear uno nuevo
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()

    # Crear fila con info del experimento
    row = {
        "parking_id": parking_id,
        "modelo": modelo,
        "tipo": tipo,
    }

    # Agregar métricas finales
    row.update(final_metrics)

    # Agregar hiperparámetros si hay
    if best_params:
        for k, v in best_params.items():
            row[f"param_{k}"] = v

    # Agregar al DataFrame
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # Guardar de nuevo
    df.to_csv(csv_path, index=False)
