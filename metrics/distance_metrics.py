import torch
import torchvision.transforms as T
from torchvision.datasets import MNIST, CIFAR10
from datasets.celeba import CelebA1000, CelebAAttr
from datasets.custom_subset import SingleClassSubset
from datasets.facescrub import FaceScrub
from datasets.stanford_dogs import StanfordDogs
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from torchvision.transforms.transforms import Resize
from utils.stylegan import create_image
from tqdm import tqdm

class DistanceEvaluation():

    def __init__(self, model, generator, img_size, center_crop_size, dataset,
                 seed):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset_name = dataset['type']
        self.dataset_path = dataset['path']
        self.model = model
        self.center_crop_size = center_crop_size
        self.img_size = img_size
        self.seed = seed
        self.train_set = self.prepare_dataset()
        self.generator = generator

    def prepare_dataset(self):
        # Build the datasets
        if self.dataset_name == 'facescrub':
            transform = T.Compose([
                T.Resize((self.img_size, self.img_size), antialias=True),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            train_set = FaceScrub(group='all',
                                  train=True,
                                  transform=transform,
                                  split_seed=self.seed)
        elif self.dataset_name == 'celeba_identities':
            transform = T.Compose([
                T.Resize(self.img_size, antialias=True),
                T.ToTensor(),
                T.CenterCrop((self.img_size, self.img_size)),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            train_set = CelebA1000(train=True,
                                   transform=transform,
                                   split_seed=self.seed,
                                   root=self.dataset_path)
        elif self.dataset_name == 'celeba_attr':
            transform = T.Compose([
                T.Resize(self.img_size, antialias=True),
                T.ToTensor(),
                T.CenterCrop((self.img_size, self.img_size)),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            train_set = CelebAAttr(train=True,
                                        transform=transform,
                                        split_seed=self.seed,
                                        root=self.dataset_path)
        elif self.dataset_name == 'mnist':
            transform = T.Compose([
                               T.ToTensor(),
                               T.Normalize(
                                 (0.1307,), (0.3081,))
                             ])
            train_set = MNIST(self.dataset_path, train=True, download=True,
                              transform=transform)  
        elif self.dataset_name == 'cifar10':
            transform = T.Compose([
                T.Resize(self.img_size, antialias=True),
                T.ToTensor(),
                T.CenterCrop((self.img_size, self.img_size)),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            train_set = CIFAR10(self.dataset_path, train=True, download=True, transform=transform)
        elif 'stanford_dogs' in self.dataset_name:
            transform = T.Compose([
                T.Resize((self.img_size, self.img_size), antialias=True),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            train_set = StanfordDogs(train=True,
                                     cropped=True,
                                     transform=transform,
                                     split_seed=self.seed,
                                     root=self.dataset_path)
        else:
            raise RuntimeError(
                f'{self.dataset_name} is no valid dataset name. Chose of of [facescrub, celeba_identities, celeb_attr, mnist, cifar10, stanford_dogs].'
            )

        return train_set

    def compute_dist(self, w, targets, batch_size=64, rtpt=None):
        self.model.eval()
        self.model.to(self.device)
        target_values = set(targets.cpu().tolist())
        smallest_distances = []
        mean_distances_list = [['target', 'mean_dist']]
        print("###Target values:",len(target_values), target_values)
        for target in target_values:
            target_subset = SingleClassSubset(self.train_set,
                                    target_class=target)
            print("$$$$$$$$$ cls", target,"-->",len(target_subset))
        for step, target in tqdm(enumerate(target_values),desc="Compute dist",total=len(target_values)):
            mask = torch.where(targets == target, True, False)
            w_masked = w[mask]
            target_subset = SingleClassSubset(self.train_set,
                                              target_class=target)
            print("###Target subset:",len(target_subset),target)
            target_embeddings = []
            for x, y in DataLoader(target_subset, batch_size):
                with torch.no_grad():
                    x = x.to(self.device)
                    outputs = self.model(x)
                    target_embeddings.append(outputs.cpu())
            try:
                print("target embeddings -->",len(target_embeddings))
                print("target embeddings -->",target_embeddings[0].shape)
            except:
                print("SKip ",target)
                continue
            attack_embeddings = []
            for w_batch in DataLoader(TensorDataset(w_masked),
                                      batch_size,
                                      shuffle=False):
                with torch.no_grad():
                    w_batch = w_batch[0].to(self.device)
                    imgs = create_image(w_batch,
                                        self.generator,
                                        crop_size=self.center_crop_size,
                                        resize=(self.img_size, self.img_size),
                                        device=self.device,
                                        batch_size=batch_size)
                    imgs = imgs.to(self.device)
                    outputs = self.model(imgs)
                    attack_embeddings.append(outputs.cpu())
                    
            print("attack embeddings -->",len(attack_embeddings))
            print("attack embeddings -->",attack_embeddings[0].shape)
            
            target_embeddings = torch.cat(target_embeddings, dim=0)
            attack_embeddings = torch.cat(attack_embeddings, dim=0)
            distances = torch.cdist(attack_embeddings, target_embeddings,
                                    p=2).cpu()
            distances = distances**2
            distances, _ = torch.min(distances, dim=1)
            smallest_distances.append(distances.cpu())
            mean_distances_list.append([target, distances.cpu().mean().item()])

            if rtpt:
                rtpt.step(
                    subtitle=
                    f'Distance Evaluation step {step} of {len(target_values)}')

        smallest_distances = torch.cat(smallest_distances, dim=0)
        return smallest_distances.mean(), mean_distances_list

    def find_closest_training_sample(self, imgs, targets, batch_size=64):
        self.model.eval()
        self.model.to(self.device)
        closest_imgs = []
        smallest_distances = []
        resize = Resize((self.img_size, self.img_size), antialias=True)
        for img, target in zip(imgs, targets):
            img = img.to(self.device)
            img = resize(img)
            if torch.is_tensor(target):
                target = target.cpu().item()
            target_subset = SingleClassSubset(self.train_set,
                                              target_class=target)
            if len(img) == 3:
                img = img.unsqueeze(0)
            target_embeddings = []
            with torch.no_grad():
                # Compute embedding for generated image
                output_img = self.model(img).cpu()
                # Compute embeddings for training samples from same class
                for x, y in DataLoader(target_subset, batch_size):
                    x = x.to(self.device)
                    outputs = self.model(x)
                    target_embeddings.append(outputs.cpu())
            # Compute squared L2 distance
            target_embeddings = torch.cat(target_embeddings, dim=0)
            distances = torch.cdist(output_img, target_embeddings, p=2)
            distances = distances**2
            # Take samples with smallest distances
            distance, idx = torch.min(distances, dim=1)
            smallest_distances.append(distance.item())
            closest_imgs.append(target_subset[idx.item()][0])
        return closest_imgs, smallest_distances
