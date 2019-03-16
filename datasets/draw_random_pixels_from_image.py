import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

def draw_random_pixels_from_image(word_image="TEXT", n_points=1000):
    from pandas import cut
    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    sp = fig.add_subplot(111)
    sp.axis('off');
    sp.text(0.5, 0.4, word_image, transform=sp.transAxes, ha='center', va='center', weight='bold', size=80)
    #save the image
    fig.savefig("_temp.png")
    #close the figure
    plt.close(fig); plt.clf(); del fig;
    #read in the image
    pl = imread("_temp.png")
    #from a 3D panel to a 2D matrix
    mx = pl[::-1,:,0].T    # the slicer ::-1 and Transpose is needed, otherwise the text will be upside down
    #make an m x 2 matrix of random numbers (0,1)
    M = np.random.rand(n_points*20, 2)
    #turn these random numbers into random index-integers
    nx = np.int16(M * mx.shape).T
    #subset those pixels that are less than 1 (1 = white)
    mask = mx[(*nx,)] < 1     # same as mx[nx[0],nx[1]]
    #use the mask on the random index-integers matrix and limit it to n_points
    X = M[mask][:n_points]
    #strech the image x-wise by multiplying the x-values by the height/width ratio
    X[:,0] *= mx.shape[0]/mx.shape[1]  # strech the image x-wise
    #sort the index matrix
    nx = np.argsort(X[:,0])
    X,y = X[nx], X[nx,0]
    #bin the x-values to make a descrete target array (the result will not be 100% accurate though)
    y = cut(y, bins=len(word_image), labels=range(len(word_image))).tolist()
    return X,y





def main():

    X,y = draw_random_pixels_from_image("HELLO")

    print(X)
    print(y)

    plt.scatter(*X.T, c=y,cmap='rainbow')   #plt.scatter(*X.T, c=y,cmap=plt.cm.get_cmap('rainbow', 5))
    plt.axis('off')
    plt.show()

if __name__=="__main__":main()