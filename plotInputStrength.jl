
# this plot Fig 7.3 of the thesis : it shows the strength of the input for input scattered around the maze to the neuron defined by "indexneuroncenter" 
using PyPlot
using Colors
using PyCall
@pyimport numpy as np
@pyimport matplotlib.patches as patch
ioff()
# plot the scalar product between 2 neurons of the space 
indexneuroncenter=20; # index of the circle location 
# establish the grid of points in the pool
# draw a small circle around the iput location to check that activity is centered around input 
radiuscircle=2.;
theta=0:0.001:2*pi+0.001;
scalarstate=[neuronsencodingvector[indexneuroncenter,:]'*matrixneurons[i,1:q] for i=1:numberofstates]
fig2 = figure("Line Collection Example")
ax = PyPlot.axes(xlim = (-R-R/100,R+R/100),ylim=(-R-R/100,R+R/100))
# plot circle 
plot(R*cos.(theta),R*sin.(theta),color="dimgray",zorder=1,lw=2)
plot(neuronscoordinates[:,indexneuroncenter][1].+radiuscircle.*cos.(theta),neuronscoordinates[:,indexneuroncenter][2].+radiuscircle.*sin.(theta),linewidth=2,color="r")
scatter(coordstates[1,:].-1/2,coordstates[2,:].-1/2,c=scalarstate);#,color="r") # add plot of the input 

colorbar()
ax[:set_axis_off]()
savefig("Run_sigma$(sigma)q$(q)freeparam$(freeparam)/Input$(indexneuroncenter).png")
close()
ioff()

indexneuroncenter=100; # index of the circle location 
# establish the grid of points in the pool
# draw a small circle around the iput location to check that activity is centered around input 
radiuscircle=2.;
theta=0:0.001:2*pi+0.001;
scalarstate=[neuronsencodingvector[indexneuroncenter,:]'*matrixneurons[i,1:q] for i=1:numberofstates]
fig2 = figure("Line Collection Example")
ax = PyPlot.axes(xlim = (-R-R/100,R+R/100),ylim=(-R-R/100,R+R/100))
# plot circle 
plot(R*cos.(theta),R*sin.(theta),color="dimgray",zorder=1,lw=2)
plot(neuronscoordinates[:,indexneuroncenter][1].+radiuscircle.*cos.(theta),neuronscoordinates[:,indexneuroncenter][2].+radiuscircle.*sin.(theta),linewidth=2,color="r")
scatter(coordstates[1,:].-1/2,coordstates[2,:].-1/2,c=scalarstate);#,color="r") # add plot of the input 

colorbar()
ax[:set_axis_off]()
savefig("Run_sigma$(sigma)q$(q)freeparam$(freeparam)/Input$(indexneuroncenter).png")
close()
ioff()

indexneuroncenter=300; # index of the circle location 
# establish the grid of points in the pool
# draw a small circle around the iput location to check that activity is centered around input 
radiuscircle=2.;
theta=0:0.001:2*pi+0.001;
scalarstate=[neuronsencodingvector[indexneuroncenter,:]'*matrixneurons[i,1:q] for i=1:numberofstates]
fig2 = figure("Line Collection Example")
ax = PyPlot.axes(xlim = (-R-R/100,R+R/100),ylim=(-R-R/100,R+R/100))
# plot circle 
plot(R*cos.(theta),R*sin.(theta),color="dimgray",zorder=1,lw=2)
plot(neuronscoordinates[:,indexneuroncenter][1].+radiuscircle.*cos.(theta),neuronscoordinates[:,indexneuroncenter][2].+radiuscircle.*sin.(theta),linewidth=2,color="r")
scatter(coordstates[1,:].-1/2,coordstates[2,:].-1/2,c=scalarstate);#,color="r") # add plot of the input 

colorbar()
ax[:set_axis_off]()
savefig("Run_sigma$(sigma)q$(q)freeparam$(freeparam)/Input$(indexneuroncenter).png")
close()