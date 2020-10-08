from . import functional as F


class FilterVisualization:

    def __init__(self, model, index=None, verbose=0):
        self.model = model
        self.index = index
        self.verbose = verbose

    def __call__(self):
        return F.visualize_filters(self.model, self.index, self.verbose)


class FeatureMapVisualization:

    def __init__(self, model, index=None, verbose=0):
        self.model = model
        self.index = index
        self.verbose = verbose

    def __call__(self, tensor):
        return F.visualize_feature_maps(self.model, tensor, self.index, self.verbose)


class FeatureExtractor:

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in dict(self.model.named_children()).items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelFeatureExtractor:

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in dict(self.model.named_children()).items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)
        return target_activations, x


class GradCAM:

    def __init__(self, model, feature_module, target_layer_names):
        self.model = model
        self.model.eval()
        self.feature_module = feature_module
        self.model_extractor = ModelFeatureExtractor(self.model, self.feature_module, target_layer_names)

    def forward(self, x):
        return self.model(x)

    def __call__(self, batch, index=None):
        return F.get_gradient_class_activation_maps(self.model, batch, index, self.feature_module, self.model_extractor)


class CAMVisualization:

    def __init__(self, batch, mask):
        self.batch = batch.cpu().detach()
        self.mask = mask.cpu().detach()

    def __call__(self):
        F.view_gradient_class_activation_maps(self.batch, self.mask)

    def save(self, filename_prefix='visualized_activation_map'):
        F.save_gradient_class_activation_maps(self.batch, self.mask, filename_prefix)
