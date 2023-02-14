import torch


class AverageMeter(object):
    def __init__(self):
        self.reset()
        self.history = {"avg": [], "sum": [], "cnt": []}

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def save(self):
        self.history["avg"].append(self.avg)
        self.history["sum"].append(self.sum)
        self.history["cnt"].append(self.cnt)


class LinearRegressor(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.model = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=False)

    def forward(self, x):
        return self.model(x)


class OutlierModel(torch.nn.Module):
    def __init__(self, in_features):
        """
        in_features: dimension of x + 1 (for y)
        """
        super().__init__()
        self.outlier_linear_model = torch.nn.Linear(in_features=in_features, out_features=1)
        self.outlier_activation = torch.nn.Sigmoid()

    def forward(self, x):
        s = self.outlier_linear_model(x).squeeze()
        return self.outlier_activation(s)


def train(train_loader, regressor, loss_criterion, optimizer, train_loss_meter):
    train_results = {"points": [], "outputs": [], "label": []}
    for point, value, _ in train_loader:
        point = torch.unsqueeze(point, -1)
        # value = torch.unsqueeze(value, -1)
        optimizer.zero_grad()
        output = regressor(point)

        for i in range(point.shape[0]):
            train_results["points"].append(point[i].item())
            train_results["label"].append(value[i].item())
            train_results["outputs"].append(output[i].item())

        loss = loss_criterion(output, torch.unsqueeze(value, -1))
        loss.backward()
        optimizer.step()
        train_loss_meter.update(loss.item())
    train_loss_meter.save()
    train_loss_meter.reset()

    return train_results


def validate(val_loader, regressor, loss_criterion, val_loss_meter):
    val_results = {"points": [], "outputs": [], "label": []}
    for point, value in val_loader:
        with torch.no_grad():
            point = torch.unsqueeze(point, -1)
            output = regressor(point)
            val_results["points"].append(point.item())
            val_results["label"].append(value.item())
            val_results["outputs"].append(output.item())
            loss = loss_criterion(output, torch.unsqueeze(value, -1))
            val_loss_meter.update(loss.item())
    val_loss_meter.save()
    val_loss_meter.reset()

    return val_results
