def fine_tuned ( model, dataset_train, dataset_val, config )
    # Training - Stage 1
    print("Training network heads")
    model.train(dataset_train, dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=40,
        layers='heads'
    )

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=120,
        layers='4+'
    )

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
        learning_rate=config.LEARNING_RATE / 10,
        epochs=150,
        layers='all'
    )
