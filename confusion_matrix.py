def createConfusionMatrix(loader):
    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, labels in loader:
        output = Net(inputs) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

    print(f"y_pred: {y_pred}")
    print("\v\v\v")
    print(f"y_true: {y_true}")

    # constant for classes -- config.CLASSES
    classes = config.CLASSES

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    # Create Heatmap
    plt.figure(figsize=(12, 7))
    plt.savefig('output1.png')
    return sn.heatmap(df_cm, annot=True, fmt=".2f", xticklabels=classes, yticklabels=classes).get_figure()
    #sn.heatmap(df_cm, annot=True).get_figure()

    
def createConfusionMatrixForEachClass(data_loader, model):
    # Define a custom function to create a confusion matrix for each class
    # Set the model in evaluation mode
    model.eval()
    
    # Initialize variables to store the confusion matrices
    num_classes = len(config.CLASSES)
    all_confusion_matrices = np.zeros((num_classes, num_classes))
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # Compute the confusion matrix for this batch
            batch_confusion_matrix = confusion_matrix(labels.numpy(), predicted.numpy(), labels=np.arange(num_classes))
            
            # Add the batch confusion matrix to the overall confusion matrix
            all_confusion_matrices += batch_confusion_matrix
    
    return all_confusion_matrices

if __name__ == '__main__':
    pass