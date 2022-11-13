import torch
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

import model
import dataset
import util

def experiment(
        data: util.Dataset,
        epoch: int=100,
        lr: float=1e-3,
        device: torch.device=torch.device('cpu')
) -> float:
    shadow_train, shadow_out, target_train, target_out, normalization = util.get_dataset(data)

    # Train the target model M on target_train
    M = model.TargetModel(
        epochs=epoch,
        dataset_type=data,
        device=device,
        lr=lr
    )
    target_train_loader = dataset.get_dataloader(
        d=target_train,
        normalize=normalization,
        shuffle=True
    )
    target_out_loader = dataset.get_dataloader(
        d=target_out,
        normalize=normalization,
    )
    M.train(target_train_loader)
    M_positive_samples = M.inference(target_train_loader)
    M_negative_samples = M.inference(target_out_loader)
    dataset.dump_D(M_positive_samples, M_negative_samples, 'target_model')

    # Derive the ground truth (test) dataset of A on M's outputs
    M_positive_features = dataset.feature_vector(M_positive_samples)
    M_negative_features = dataset.feature_vector(M_negative_samples)
    M_feature_labels = np.concatenate([np.ones(len(M_positive_features), dtype=int),
                                       np.zeros(len(M_negative_features), dtype=int)])
    M_features = np.concatenate([M_positive_features, M_negative_features])
    attack_test_loader = dataset.get_dataloader(
        d=(M_features, M_feature_labels),
    )

    # Train the shadow model S on shadow_train
    S = model.ShadowModel(
        epochs=epoch,
        dataset_type=data,
        device=device,
        lr=lr
    )
    shadow_train_loader = dataset.get_dataloader(
        d=shadow_train,
        normalize=normalization,
        shuffle=True
    )
    shadow_out_loader = dataset.get_dataloader(
        d=shadow_out,
        normalize=normalization,
    )
    S.train(shadow_train_loader)
    S_positive_samples = S.inference(shadow_train_loader)
    S_negative_samples = S.inference(shadow_out_loader)
    dataset.dump_D(S_positive_samples, S_negative_samples, 'shadow_model')

    # Derive the training dataset of A on S's outputs
    S_positive_features = dataset.feature_vector(S_positive_samples)
    S_negative_features = dataset.feature_vector(S_negative_samples)
    S_feature_labels = np.concatenate([np.ones(len(S_positive_features), dtype=int),
                                       np.zeros(len(S_negative_features), dtype=int)])
    S_features = np.concatenate([S_positive_features, S_negative_features])
    attack_train_loader = dataset.get_dataloader(
        d=(S_features, S_feature_labels),
        shuffle=True
    )

    A = model.AttackModel(
        epochs=10,
        lr=0.01,
        device=device,
    )
    A.train(attack_train_loader)
    pred = A.predict(attack_test_loader)

    return sum(pred == M_feature_labels) / len(pred)

if __name__ == '__main__':
    device = torch.device('cuda:0')

    acc = experiment(util.Dataset.CIFAR, device=device, epoch=50, lr=1e-3)
    print(f'{util.Dataset.CIFAR.name} {acc:.3f}')

    acc = experiment(util.Dataset.MNIST, device=device, epoch=20)
    print(f'{util.Dataset.MNIST.name} {acc:.3f}')
