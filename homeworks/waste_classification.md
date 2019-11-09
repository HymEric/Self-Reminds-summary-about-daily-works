## This is a homework in Artificial intelligence to sort garbage.

### Most important: data balance and data pre-processing and the proper model!

### Condition
- Data: 
  - [six classes of waste classification in official website](https://momodel.cn/explore/5d411ace1afd9427c236eab5?type=dataset) which has 2507 images, each class details:
  
    |No.|Chinese Name|English Name|Size of images|
    |----|----|----|----|
    |1|玻璃|glass|497 |
    |2|纸|paper|590 |
    |3|硬纸板|cardboard| 400 |
    |4|塑料|plastic| 479 |
    |5|金属|metal| 407 |
    |6|一般垃圾|trash| 134 |
    
  - But there are large dataset from [trashnet](https://github.com/garythung/trashnet) which has more 20 images. See it for detail.

- Task:
  - Select a appropriate model to train a deep model for best acuracy by testing in [Mo](https://momodel.cn/)

### Pipeline
- First, we should check the dataset, we use official datasets which has  2507 images. From above tables we can see this datasets is unbalanced because the trash class is almost quarter (1/4) of other classes whihc has only 134 images. For example, the paper is 590 which is more larger tham trash. So first step we should apply **data balance method** to this datases. For example, we just copy trash dataset for more two times, thus it has 411 images for easily. And do **data augmentation** together in the training stage.

**Notes: If you do not do the data balance, you maybe find the training accuracy and validation accuracy is weird, like train:90, val:50 Mo-test:5/10 or train:70, val:90, Mo-test:7/10. Also, it maybe because over-fitting or under-fitting. But ths datasets play an significant role in the real project. And I just want to remind you and me to keep it in mind!**

- Split datasets: choose enough dataset for vlaisation. For example, I choose 100 images each of classes to do validation in the training stage.
- Select one model to taining and watch the loss and accuracy changes.

### Records of mime
- I do data balance of trash class by just copy trash dataset for more two times, thus it has 411 images for easily, and do data augmentation in the training stage, like bellow. I choose ResNet50-pretrained to tain this model and it can reach 94% accuracy in the validation stage. But it pass 9/10 in the Mob workspace. Oh, the bad cases always just only one!
```python
# Data loading code in pytorch
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(333),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
```
- Later, I analyze a lot of condition and it take me many time. First, I check wich class is hard to train, and I found the paper, plastic and metal is not 100% accuracy. Perhaps their images type like color, shape, contrast, illumination is more rice. Surely, the common way to solve this problem is set calss weights. I set the *class_weights* to help hard case converge, like belloe. But it's not very good and the accuracy is also about 95%. Pass 9/10. Oh,no. The last bad case is terrible!
```python
# define loss function (criterion) in pytorch
    # adding class_weight    dict_key={0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}
    weights = [1.0,1.0,1.25,1.5,1.25,1.0]
    weights=weights+[1.0]*994 # because I use default numn_classes=1000, you can change it to 6
    class_weights = torch.FloatTensor(weights).cuda(args.gpu)
    criterion = nn.CrossEntropyLoss(class_weights).cuda(args.gpu)
```
- Moreover, I changed several models (resnet18, resnet50, and resnet101) setting based on ResNet like batch_size, learing_rate and whether to use pre-trained mode, even the gray-mode training. But all of these can not address the last hating one bad case! It is still 9/10!
- At last, I realized that I maybe ought to change another sota model to try. The last bad case maybe for the leaning ability is not enough with resnet model. Therefore, I use Xception to re-train with above data balance and data pre-processing. Congratulation! It worked! Do you think of anything?  Yes, yes! The NAS (Neural architecture search), maybe it is the best great method for deep learning!

**Notes: The model architecture is the second important factor after proper datasets operation!**

