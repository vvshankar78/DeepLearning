def train(model, device,trainloader,optimizer,criterion,epoch):
  running_loss = 0.0

  for i, data in enumerate(trainloader, 0):
    # get the inputs
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)


    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    if i % 2000 == 1999:    # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss / 2000))
        running_loss = 0.0