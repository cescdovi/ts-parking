from TrainOneEpoch import train_one_epoch
from TrainOneEpoch import validate_one_epoch
import time

def fit_simple(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    metrics,
    scheduler=None,
    num_epochs=10,
    device="cpu",
    early_stopper = None,

):
    """
    Entrena un modelo simple (debug/test) para un solo parking. 
    Devuelve el modelo entrenado, el historial y las métricas finales.
    Incluye prints detallados por época.
    """
    history = []
    model.to(device)

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            metrics=metrics.clone(),
            device=device,
            epoch=epoch
        )

        val_metrics = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            metrics=metrics.clone(),
            device=device,
            epoch=epoch,
        )

        if scheduler:
            scheduler.step()

        combined = {"epoch": epoch}
        combined.update(train_metrics)
        combined.update(val_metrics)
        history.append(combined)

         # Tiempo total de la época
        total_time = time.time() - epoch_start

        # mostrar logs con emojis para mayor claridad
        print(
            f"📊 Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_metrics['train_loss']:.4f} | "
            f"Train MAE: {train_metrics.get('MAE', 0):.2f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val MAE: {val_metrics.get('MAE', 0):.2f} | "
            f"⏱️ Time: {total_time:.1f}s"
        )

        # Early stopping
        if early_stopper is not None:
            if early_stopper(val_metrics["val_loss"], model):
                print(f"⛔ Parada temprana en epoch {epoch} por falta de mejora.")
                break

    return model, history, val_metrics
