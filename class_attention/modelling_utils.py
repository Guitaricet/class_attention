import torch
import transformers


def get_output_dim(model: transformers.PreTrainedModel):
    # it looks like Transformers changed this in some version
    # config = model.config
    # if isinstance(config, transformers.DistilBertConfig):
    #     return config.hidden_size
    return model.config.hidden_size


def validate_inputs(text_input_dict, labels_input_dict):
    if not isinstance(text_input_dict, dict):
        raise ValueError('text input should be a dict')
    if not isinstance(labels_input_dict, dict):
        raise ValueError('classes input should be a dict')

    if labels_input_dict['input_ids'].shape[0] == 1:
        raise RuntimeError(
            'batch dimension of classes tensor is the number of possible classes and cannot be equal to one'
        )

    # check that labels_input does not have duplicated
    unique_classes = torch.unique(labels_input_dict['input_ids'], dim=0)
    if unique_classes.shape[0] != labels_input_dict['input_ids'].shape[0]:
        raise ValueError('labels_input should only contain unique classes')


def maybe_format_inputs(text_input, labels_input):
    if isinstance(text_input, torch.Tensor):
        text_input = {"input_ids": text_input}

    if isinstance(labels_input, torch.Tensor):
        labels_input = {"labels_input": labels_input}

    return text_input, labels_input
