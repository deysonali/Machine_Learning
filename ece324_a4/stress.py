for folder in os.listdir(my_im_folder):
    for picture in os.listdir(my_im_folder + "/" + folder):
        ims.append(plt.imread(my_im_folder + "/" + folder + "/" + picture))
ims = np.asarray(ims)
print("shape=", np.shape(ims))
print("Image set:")
# print(ims)
mean = ims.mean(axis=(0, 1, 2))
print("My image set mean = ", mean)
std = ims.std(axis=(0, 1, 2))
print("My image set std = ", std)

transformation = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((mean[0] / 255, mean[1] / 255, mean[2] / 255),
                                                          (std[0] / 255, std[1] / 255, std[2] / 255))])
mean = 0.0
std = 0.0
train = torchvision.datasets.ImageFolder(root="./asl_images_myset", transform=transformation)
untransformed = torchvision.datasets.ImageFolder(root="./asl_images_myset", transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True, num_workers=2)