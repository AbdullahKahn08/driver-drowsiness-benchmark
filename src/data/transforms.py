from torchvision import transforms

training_transform_pipeline = transforms.Compose(transforms=[transforms.Resize((224,224)),
                                                    transforms.RandomHorizontalFlip(p=0.5),
                                                    transforms.RandomRotation([30,45]),
                                                    transforms.ColorJitter(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

testVal_transform_pipeline = transforms.Compose(transforms=[transforms.Resize((224,224)),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
                                                            