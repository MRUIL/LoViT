def get_transitionmap(self, label_list, v_id):
    transitionmap = []
    lenth = len(label_list)
    label_newest = -1
    label_indx = -1000000
    sigma = 13
    tmp_size = 3*sigma
    for i in range(lenth):
        label_now = label_list[i].item()
        if label_now != label_newest:
            label_indx = i
            if i == 0:
                label_indx = -1000000
            label_newest = label_now
        dis = i - label_indx
        if dis<= tmp_size:
            g = np.exp(-dis**2/(2 * sigma**2))
        else:
            g = 0
        transitionmap.append(g)

    label_indx = 1000000
    label_newest = -1
    
    sigma = 2
    tmp_size = 3*sigma
    for j in range(lenth):
        i = lenth -1 - j
        label_now = label_list[i].item()
        if label_now != label_newest:
            label_indx = i
            if j == 0:
                label_indx = 1000000
            label_newest = label_now
        dis = label_indx - i
        if dis<= tmp_size:
            g = np.exp(-dis**2/(2 * sigma**2))
        else:
            g = 0
        transitionmap[i] =max(transitionmap[i], g)
    
    transitionmap_image = (np.array(transitionmap)*255).astype('uint8')
    transitionmap_image = np.expand_dims(transitionmap_image,0).repeat(20,axis = 0)
    transitionmap_image = np.expand_dims(transitionmap_image,2).repeat(3,axis = 2)
    cv2.imwrite('transitionmap/'+str(v_id)+'.jpg',transitionmap_image)
    transitionmap = torch.Tensor(transitionmap).float()
    return transitionmap