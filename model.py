import torch
import torch.nn.functional as F
from torch import nn
from sklearn.decomposition import PCA

class clasifier(nn.Module):
    def __init__(self, backbone, n_dim):
        super(clasifier, self).__init__()
        self.backbone = backbone
        self.fc = nn.Linear(384, n_dim)

    def forward(self, x):
        features1 = self.backbone(x)
        features2 = self.fc(features1)
        features2 = F.normalize(features2)
        return features2

def get_dino_features(dino, images):
    features = dino.forward_features(images)
    return features

def get_feature_image(feature):
  pca = PCA(n_components=3)
  pca.fit(feature)
  pca_features = pca.transform(feature)
  pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
  pca_features = pca_features * 255
  pca_features = pca_features.reshape(56, 56, 3).astype(np.uint8)
  return pca_features