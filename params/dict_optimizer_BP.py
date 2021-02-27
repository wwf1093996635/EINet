lr = 1.0e-2

dict_ ={
    'class': 'optimizer',
    'type': 'bp',
    'optimizer_dict':{
        'type': 'sgd',
        'lr': lr
    },
    'lr_decay': {
        'method': 'linear',
        'milestones': [[0.50, 1.0],[0.70, 1.0e-1], [0.85, 1.0e-2], [0.95, 1.0e-3]],
    }
}

def interact(model_dict=None, optimizer_dict=None, trainer_dict=None, data_loader_dict=None, **kw):
    return