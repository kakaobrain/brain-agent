import nle.nethack as nh

def get_object(name, return_index=False):
    for index in range(nh.NUM_OBJECTS):
        obj = nh.objclass(index)
        if nh.OBJ_NAME(obj) == name:
            if return_index:
                return obj, index
            else:
                return obj