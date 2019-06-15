# Label names
LABELS = {0: "Anger", 1: "Disgust", 2: "Fear",
          3: "Hapiness", 4: "Neutral", 5: "Sadness", 6: "Surprise"}


MODELS = {
    "ce-xception-512-256": {
        "filename": "ce-xception-512-256.h5",
        "url": "https://github.com/deepexpressions/models/blob/master/models/ce-xception-512-256/model.h5?raw=true",
        "info": """
        Model: "ce-xception-512-256"
        ________________________________________________________________________________
        Parameter                   Value  
        ================================================================================
        ConvNet                     Xception
        Dataset                     Compound Facial Expressions Database
        NN-Layers                   [512, 256]
        Input shape                 [None, 128, 128, 3]
        ________________________________________________________________________________
        Optimizer                   Adam      
        Learning rate               1e-3
        Epochs                      30      
        Epochs (fn)                 15
        ConvNet layers (fn)         100+
        ================================================================================
        * fn = fine-tunning

        ================================================================================
        https://github.com/deepexpressions/models/tree/master/models/ce-xception-512-256
        ================================================================================
        """
    },
}