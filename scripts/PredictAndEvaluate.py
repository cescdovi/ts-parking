import torch
import os
import json
from Saving_and_Paths import get_experiment_path
def predict_and_evaluate(model, dataloader, metrics, device="cpu", parking_id=None,
                         modelo=None, tipo=None, base_dir="../test_predictions"):
    """
    Predice sobre el dataloader, calcula métricas y guarda resultados en carpeta estructurada por parking.

    Args:
        model (nn.Module): Modelo entrenado.
        dataloader (DataLoader): Conjunto de test.
        metrics (torchmetrics.MetricCollection): Métricas para calcular.
        device (str): "cpu" o "cuda".
        parking_id (int or str): ID del parking (para crear carpeta).
        modelo (str): Nombre del modelo ("lstm", "rnn", etc.).
        tipo (str): Tipo de experimento ("vanilla", "optuna", etc.).
        base_dir (str): Carpeta base para guardar.

    Returns:
        dict: Métricas de test.
        pd.DataFrame: DataFrame con y_true y y_pred.
    """
    model.eval()
    model.to(device)
    metrics.to(device)
    metrics.reset()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(x_batch)

            all_preds.append(y_pred.cpu())
            all_targets.append(y_batch.cpu())

    # Unir predicciones y targets
    y_pred = torch.cat(all_preds, dim=0).squeeze().numpy()
    y_true = torch.cat(all_targets, dim=0).squeeze().numpy()

    # Calcular métricas
    y_pred_tensor = torch.tensor(y_pred)
    y_true_tensor = torch.tensor(y_true)
    metrics.reset()
    metrics.update(y_pred_tensor, y_true_tensor)
    results = {k: v.item() for k, v in metrics.compute().items()}

    # Guardar resultados
    base_path = get_experiment_path(base_dir, parking_id, modelo, tipo)
    test_path = os.path.join(base_path, "test")
    os.makedirs(test_path, exist_ok=True)

    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred
    })
    df.to_csv(os.path.join(test_path, "predictions.csv"), index=False)

    with open(os.path.join(test_path, "test_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results, df
