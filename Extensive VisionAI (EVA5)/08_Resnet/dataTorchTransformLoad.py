def data_transform_and_load_function(dataset_to_load_param,batch_size_param):
  import torch
  from torchvision import datasets, transforms
  # Train Phase transformations

  if(dataset_to_load_param.upper() == 'MNIST'):
    train_transforms = transforms.Compose([
                                          #  transforms.Resize((28, 28)),
                                          #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                          transforms.RandomRotation((-5.0, 5.0), fill=(1,)), 
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,)), # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
                                          #  transforms.RandomErasing()
                                          # Note the difference between (0.1307) and (0.1307,)
                                          ])

    # Test Phase transformations
    test_transforms = transforms.Compose([
                                          #  transforms.Resize((28, 28)),
                                          #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))
                                          ])

    train = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
    test = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)       

  if(dataset_to_load_param.upper() == 'CIFAR10'):
    transforms_cifar_train = transforms.Compose([
                                          #  transforms.Resize((28, 28)),
                                          #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                          transforms.RandomHorizontalFlip(),
                                          #   transforms.RandomRotation((-5.0, 5.0), fill=(1,)), 
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              [0.49139968, 0.48215841, 0.44653091], 
                                              [0.24703223, 0.24348513, 0.26158784]), # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
                                          #  transforms.RandomErasing()
                                          # Note the difference between (0.1307) and (0.1307,)
                                          ])
    transforms_cifar_test = transforms.Compose([
                                              #  transforms.Resize((28, 28)),
                                              #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                            #   transforms.RandomRotation((-5.0, 5.0), fill=(1,)), 
                                            #   transforms.RandomHorizontalFlip(0.3),
                                              transforms.ToTensor(),
                                              transforms.Normalize(
                                                  [0.49139968, 0.48215841, 0.44653091], 
                                                  [0.24703223, 0.24348513, 0.26158784]), # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
                                              #  transforms.RandomErasing()
                                              # Note the difference between (0.1307) and (0.1307,)
                                              ])
    
    train = datasets.CIFAR10('./data', train=True, download=True, transform=transforms_cifar_train)
    test = datasets.CIFAR10('./data', train=False, download=True, transform=transforms_cifar_test)       

  SEED = 1

  # CUDA?
  cuda = torch.cuda.is_available()
  print("CUDA Available?", cuda)

  # For reproducibility
  torch.manual_seed(SEED)

  if cuda:
      torch.cuda.manual_seed(SEED)

  # dataloader arguments - something you'll fetch these from cmdprmt
  dataloader_args = dict(shuffle=True, batch_size=batch_size_param, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=batch_size_param)

  # train dataloader
  train_loader = torch.utils.data.DataLoader(train, **dataloader_args)

  # test dataloader
  test_loader = torch.utils.data.DataLoader(test, **dataloader_args)                                

  return(train,test,train_loader,test_loader)
