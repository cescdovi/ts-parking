import torch

class EarlyStoppingCallback:
    def __init__(self, patience=5, delta=0.0, verbose=False, path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path

        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss < self.best_loss - self.delta:
            # 🔁 Mejora
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"🛑 EarlyStopping: {self.counter}/{self.patience} sin mejora")
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"✅ Mejor val_loss: {val_loss:.6f}. Guardando modelo...")
        torch.save(model.state_dict(), self.path)