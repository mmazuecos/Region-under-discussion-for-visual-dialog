def get_spatial_feat(bbox, im_width, im_height):

    x_width = bbox[2]
    y_height = bbox[3]

    x_left = bbox[0]
    x_right = x_left + x_width

    y_upper = im_height - bbox[1]
    y_lower = y_upper - y_height

    x_center = x_left + 0.5*x_width
    y_center = y_lower + 0.5*y_height

    # Rescale features fom -1 to 1

    x_left = (1.*x_left / im_width) * 2 - 1
    x_right = (1.*x_right / im_width) * 2 - 1
    x_center = (1.*x_center / im_width) * 2 - 1

    y_lower = (1.*y_lower / im_height) * 2 - 1
    y_upper = (1.*y_upper / im_height) * 2 - 1
    y_center = (1.*y_center / im_height) * 2 - 1

    x_width = (1.*x_width / im_width) * 2
    y_height = (1.*y_height / im_height) * 2

    # Concatenate features
    feat = [x_left, y_lower, x_right, y_upper, x_center, y_center, x_width, y_height]

    return feat

def constrain_bbox(game, target_bbox, hist):
    objects_bbox = [game['image']['width'], game['image']['height'], 0, 0]
    # Get distractors
    distractors = [obj for obj in game['objects'] if obj['category'] == hist[0][1]]
    if len(distractors) > 1:
        # If you have more than one distractor, then crop
        # the region, else, keep the original spatial
        # information
        for o in distractors:
            # Get object bbox
            bbox = o['bbox']
            # Generate left, upper, right, lower
            objects_bbox[0] = min(objects_bbox[0], bbox[0])
            objects_bbox[1] = min(objects_bbox[1], bbox[1])
            objects_bbox[2] = max(objects_bbox[2], bbox[0] + bbox[2])
            objects_bbox[3] = max(objects_bbox[3], bbox[1] + bbox[3]) 

        # Get new image size
        obwidth = round(objects_bbox[2] - objects_bbox[0])
        obheight = round(objects_bbox[3] - objects_bbox[1])
        # Shift bounding boxes to the new size
        target_bbox[0] -= objects_bbox[0]
        target_bbox[1] -= objects_bbox[1]
        # Get spatial information
        spatial = get_spatial_feat(bbox=target_bbox, im_width=obwidth, im_height=obheight)
        return spatial 
    else:
        return None

def calculte_error(spatial, error_func):
    lower = [-1, -1, -1, -1, -1, -1, 0, 0]
    upper = [1, 1, 1, 1, 1, 1, 2, 2]

    error = []
    for i, v in enumerate(spatial):
        if v < lower[i]:
            error.append(error_func(v,lower[i]))
        elif v > upper[i]:
            error.append(error_func(v,upper[i]))
    return error
