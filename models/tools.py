
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def write_data(data, path, clear):
    with open(path, "a") as file_object:
        if clear == True:
            file_object.truncate(0)
        else:
            file_object.write("\n")
        file_object.write(data)