# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
Noise is a common issue in real-world image data, which affects performance in image analysis tasks. An autoencoder can be trained to remove noise from images, effectively learning compressed representations that help in reconstruction. The MNIST dataset (28x28 grayscale handwritten digits) will be used for this task. Gaussian noise will be added to simulate real-world noisy data.

## DESIGN STEPS

### STEP 1:
Import necessary libraries including PyTorch, torchvision, and matplotlib.
### STEP 2:
Load the MNIST dataset with transforms to convert images to tensors.
### STEP 3:
Add Gaussian noise to training and testing images using a custom function.
## STEP 4:
Define the architecture of a convolutional autoencoder:

Encoder: Conv2D layers with ReLU + MaxPool

Decoder: ConvTranspose2D layers with ReLU/Sigmoid
## STEP 5:
Initialize model, define loss function (MSE) and optimizer (Adam).
## STEP 6:
Train the model using noisy images as input and original images as target.
## STEP 7:
Visualize and compare original, noisy, and denoised images.
### Name: SUSITHRA.B
### Register Number:212223220113

## PROGRAM
```
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()  # set the model to training mode

    for epoch in range(epochs):
        running_loss = 0.0

        for data in loader:
            # noisy_imgs, clean_imgs = data  # (noisy_input, ground_truth)
            # Instead of using the class labels (clean_imgs), 
            # use the original images as the target for reconstruction.
            images, _ = data  # images are the original clean images
            
            images = images.to(device)
            noisy_imgs = add_noise(images).to(device)  # Add noise to the input images

            # Forward pass
            outputs = model(noisy_imgs)

            # Calculate the loss between the model's output and the original images
            loss = criterion(outputs, images)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * noisy_imgs.size(0)

        avg_loss = running_loss / len(loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# Evaluate and visualize
def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name:   SUSITHRA.B          ")
    print("Register Number:  212223220113              ")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()



# Run training and visualization
train(model, train_loader, criterion, optimizer, epochs=5)
visualize_denoising(model, test_loader)
```
## OUTPUT

### Model Summary
![image](https://github.com/user-attachments/assets/112de91a-1ba0-42ad-aab5-640678687842)


### Original vs Noisy Vs Reconstructed Image

![image](https://github.com/user-attachments/assets/b5f866f8-aeb6-4ebb-98c4-7b039859c16c)
![image](https://github.com/user-attachments/assets/4f6b6473-e76d-4a9d-ab2b-f7b759ec708e)


## RESULT
The convolutional autoencoder was successfully trained to denoise MNIST digit images. The model effectively reconstructed clean images from their noisy versions, demonstrating its capability in feature extraction and noise reduction.
