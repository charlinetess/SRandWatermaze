# This produces a plot showing the scalar product between every neuron and the neuron defined as "indexneuroncenter"

using PyPlot
using Colors

@pyimport numpy as np
@pyimport matplotlib.patches as patch

# plot the scalar product between 2 neurons of the space 


indexneuroncenter=20; # index of the circle location 
# establish the grid of points in the pool

# draw a small circle around the iput location to check that activity is centered around input 
radiuscircle=2.;


theta=0:0.001:2*pi;
using PyPlot
using PyCall

scalarneurons=[neuronsencodingvector[indexneuroncenter,:]'*neuronsencodingvector[i,:] for i=1:numberofneurons]

fig2 = figure("Line Collection Example")
ax = PyPlot.axes(xlim = (-R-R/100,R+R/100),ylim=(-R-R/100,R+R/100))

# plot circle 
plot(R*cos.(theta),R*sin.(theta),color="dimgray",zorder=1,lw=2)
plot(neuronscoordinates[:,indexneuroncenter][1].+radiuscircle.*cos.(theta),neuronscoordinates[:,indexneuroncenter][2].+radiuscircle.*sin.(theta),linewidth=2,color="r")
# xneurons=zeros(numberofstates,numberofstates);
# xneurons[indexneurons,indexneurons]=X[indexneurons]

# pcolormesh(X,Y,testplotstate)
# scatter()
# scatter(coordstates[1,:].-1/2,coordstates[2,:].-1/2,c=colorstates./255)
scatter(neuronscoordinates[1,:].-1/2,neuronscoordinates[2,:].-1/2,c=scalarneurons./255);#,color="r") # add plot of the input 

colorbar()
ax[:set_axis_off]()



# savefig("Eigenvectortest$(indexneuroncenter).png")


show()

# 	