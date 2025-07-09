```python
Algorithm: Classification of Medicine Pictures
input:train_dataset,val_dataset,test_dataset
output:trained model
Initialize: CNNmodel
            scheduler:ReduceLROnPlateau
            criterion:CrossEntropyLoss
            optimizer:Adam
for epoch=0 to epochs do
    for image,label in train_loader do
        output ← CNNmodel(image)
        loss ← criterion(output,label)
        correct ← if predicted == label
        backward
        update parameters ← optimizer
    end for
    val_acc,val_loss ← evaluate(val_loader)
    lr ← scheduler(val_acc)
end for
return model

```