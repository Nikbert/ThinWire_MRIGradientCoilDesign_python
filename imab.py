import numpy as np
import matplotlib.pyplot as plt

def imab(data, clims=None, bSep=False, noCols=None):
    if noCols is None:
        noCols = []
    if clims is None:
        clims = []

    if data.shape[2] == 1:
        data = np.squeeze(data)

    if data.shape[3] > 20:
        print("Dataset more than 20 time points, could be difficult to see what's going on...!")

    data = np.transpose(data, (1, 0, 2, 3))
    if np.iscomplexobj(data):
        data = np.abs(data)
        print("Data were complex - absolute values are shown")

    if bSep and data.shape[2] > 1:
        dims = data.shape
        if clims == []:
            clims = [np.min(data), np.max(data)]

        if data.shape[3] == 1:
            if noCols is None:
                noCols = min(5, np.ceil(dims[2] / 2))
            noRows = np.ceil(dims[2] / noCols).astype(int)
            iPlots = np.reshape(np.arange(noCols * noRows, 0, -1), (noRows, noCols)).T
            iPlots = np.fliplr(iPlots)

            fig, axs = plt.subplots(noRows, noCols)
            for iR in range(noRows):
                for iC in range(noCols):
                    if noRows == 1:
                        ax = axs[iC]
                    else:
                        ax = axs[iR, iC]
                    if iPlots[iR, iC] > dims[2]:
                        ax.axis('off')
                    else:
                        hndl = ax.imshow(data[:, :, iPlots[iR, iC]], cmap='viridis', clim=clims)
                        ax.set_aspect('equal')
                        ax.set_xticks([])
                        ax.set_yticks([])
            plt.tight_layout()

        else:
            noRows = data.shape[3]
            noCols = data.shape[2]
            fig, axs = plt.subplots(noRows, noCols)
            for iR in range(noRows):
                for iC in range(noCols):
                    if noRows == 1:
                        ax = axs[iC]
                    else:
                        ax = axs[iR, iC]
                    hndl = ax.imshow(data[:, :, iC, iR], cmap='viridis', clim=clims)
                    ax.set_aspect('equal')
                    ax.set_xticks([])
                    ax.set_yticks([])
            plt.tight_layout()

    else:
        if data.shape[3] == 1:
            if data.shape[2] > 1:
                if noCols is None:
                    noCols = min(5, np.ceil(data.shape[2] / 2))
                outIm = makemosaic(data, noCols)
            else:
                outIm = data
            if clims == []:
                hndl = plt.imshow(outIm, cmap='viridis')
            else:
                hndl = plt.imshow(outIm, cmap='viridis', clim=clims)
            plt.gca().set_aspect('equal')
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
        else:
            if clims == []:
                hndl = plt.imshow(imcat(imcat(data, 2, 3), 1, 4, 0), cmap='viridis')
            else:
                hndl = plt.imshow(imcat(imcat(data, 2, 3), 1, 4, 0), cmap='viridis', clim=clims)
            plt.gca().set_aspect('equal')
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])

    plt.gca().invert_yaxis()

    if 'hndl' in locals():
        return hndl


def makemosaic(im, MaxN=5):
    """
    Make a mosaic image for displaying the image.
    Converts a 3D image 'im' to a mosaic 2D image 'imall'.
    If 'im' is 4D, 'im(:, :, :, 1)' will be used.
    
    Args:
        im (ndarray): 3D or 4D image
        MaxN (int, optional): The number of columns. Default is 5.
        
    Returns:
        imall (ndarray): Mosaic 2D image
    
    Examples:
        imall = makemosaic(im, 10)
        'imall' will be a 2x10 image of size 64x64, where size(imall) = 128 x 640
    """
    if MaxN is None:
        MaxN = 5
    
    im = np.squeeze(im)
    dim = im.shape
    
    if len(dim) < 2:
        raise ValueError('Input is 1D or 2D signal')
    elif len(dim) == 4:
        im = np.squeeze(im[:, :, :, 0])
        print('4D: TimePoint 1 was used')
    elif len(dim) > 4:
        raise ValueError('5D or higher dimensions are not supported')
    
    Nrow = np.ceil(dim[2] / MaxN).astype(int)
    Rcol = MaxN - (dim[2] % MaxN) % MaxN

    if dim[2] <= MaxN:
        imall = np.reshape(im, (dim[0], dim[1] * dim[2]))
        imall = np.concatenate((imall, np.zeros((dim[0], dim[1] * Rcol))), axis=1)
    else:
        imall = np.reshape(im[:, :, :MaxN], (dim[0], dim[1] * MaxN))
        for ii in range(1, Nrow - 1):
            temp = np.reshape(im[:, :, (ii - 1) * MaxN:ii * MaxN], (dim[0], dim[1] * MaxN))
            imall = np.concatenate((imall, temp), axis=0)
        temp = np.reshape(im[:, :, (Nrow - 1) * MaxN:], (dim[0], dim[1] * (MaxN - Rcol)))
        temp = np.concatenate((temp, np.zeros((dim[0], dim[1] * Rcol))), axis=1)
        imall = np.concatenate((imall, temp), axis=0)

    return imall


def imcat(data, outdim, indim, updown=1):
    """
    Concatenate 'data' along the 'indim' dimension and add it to the 'outdim' dimension.
    
    Args:
        data (ndarray): Input data
        outdim (int): Output dimension to add the concatenated data
        indim (int): Input dimension along which to concatenate the data
        updown (bool, optional): Concatenate data in upward (1) or downward (0) direction. Default is 1.
    
    Returns:
        out (ndarray): Concatenated output data
        
    Note:
        The input data 'data' should have at least 3 dimensions.
    """
    if updown:
        axis = outdim - 1
        temp = np.take(data, 0, axis=indim-1)

        if data.shape[indim-1] > 1:
            for i in range(1, data.shape[indim-1]):
                temp = np.concatenate((temp, np.take(data, i, axis=indim-1)), axis=outdim-1)

    else:
        axis = outdim - 1
        temp = np.take(data, -1, axis=indim-1)

        if data.shape[indim-1] > 1:
            for i in range(data.shape[indim-1] - 2, -1, -1):
                temp = np.concatenate((temp, np.take(data, i, axis=indim-1)), axis=outdim-1)

    # Remove axes and set plotting properties
    ax = plt.gca()
    ax.axis('off')
    ax.set_box('off')

    return temp

