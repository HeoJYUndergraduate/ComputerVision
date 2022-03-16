
import tensorboard
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets 
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from modelss import Discriminator, Generator, initialize_weights
from utils import gradient_penalty
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 1e-4        # Default on papers
NUM_EPOCHS = 5
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NOISE_DIM = 128

FEATURES_DISC = 32
FEATURES_GEN = 64

FEATURES_CRITIC = 64        # Default on papers
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5    # Default on papers
#WEIGHT_CLIP = 0.01      # Default on papers
LAMBDA_GP = 10


transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ])

dataset = datasets.MNIST(root = "celeb_dataset/", train = True, transform=transforms, download = True).to(DEV)

dataloader = DataLoader(dataset, BATCH_SIZE, shuffle = True)
gen= Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(DEV)
critic = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(DEV)
initialize_weights(gen)
initialize_weights(critic)

opt_gen=optim.Adam(lr=LEARNING_RATE, params=gen.parameters(), betas = (0.0, 0.9))     
# 데이터가 불안정한 경우 Momentum을 사용하지 않는 opimizer가 더 효율적이다
opt_critic=optim.RMSprop(lr=LEARNING_RATE, params=critic.parameters(), betas = (0.0, 0.9))

fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(DEV)
writer_real= SummaryWriter(f"logs/real/")
writer_fake= SummaryWriter(f"logs/fake/")
step = 0

gen.train()
critic.train()


for epoch in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.to(DEV)
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(DEV)
            fake = gen(noise)
            critic_real =critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, real, fake, device = DEV)
            loss_critic = -(torch.mean(critic_real)) + torch.mean(critic_fake) + LAMBDA_GP * gp
            critic.zero_grad()
            loss_critic.backward(retain_graph = True)
            opt_critic.step()


            # for p in critic.parameters():
            #     p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        #### Train Generator: min -E(critic(gen_fake))
        output = critic(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1