python src/train.py config/tf_efficientnetv2_b0.yaml train.epochs=3 optim.optimizer.weight_decay=1e-4
python src/train.py config/tf_efficientnetv2_b0.yaml train.epochs=5 optim.optimizer.weight_decay=1e-4

python src/train.py config/tf_efficientnetv2_b0.yaml train.epochs=3 optim.optimizer.weight_decay=1e-5
python src/train.py config/tf_efficientnetv2_b0.yaml train.epochs=5 optim.optimizer.weight_decay=1e-5
