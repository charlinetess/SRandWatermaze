using PyPlot
using PyCall
@pyimport matplotlib.patches as patch

fig2 = figure("Line Collection Example",figsize=(9,9))





colorneurons=[[74,181,57] for k=1:numberofneurons]
colorstates=[[175,252,164] for k=1:numberofstates]

scatter(coordstates[1,:].-1/2,coordstates[2,:].-1/2,c=colorstates./255)
scatter(neuronscoordinates[1,:].-1/2,neuronscoordinates[2,:].-1/2,c=colorneurons./255);#,color="r") # add plot of the input center 
# gca().grid(False) 
theta=0:pi/50:2pi+pi/50;
plot(R*cos.(theta),R*sin.(theta),ls="-",color="k")#[169/255,169/255,169/255])
# Hide axes ticks
gca().set_xlim([-R,R])
gca().set_ylim([-R,R])
gca().set_xticks([])
gca().set_yticks([])
gca().set_axis_off()
show()


# plot states with reward location and 

# theta=0:pi/50:2pi+pi/50;



# using PyPlot
# using PyCall

# fig2 = figure("Line Collection Example")
# #ax = PyPlot.axes(xlim = (-R,R),ylim=(-R,R))

# # plot circle 
# plot(R*cos.(theta),R*sin.(theta),ls="--",color=[169/255,169/255,169/255])
# scatter(coordstates[1,:],coordstates[2,:])
# # plot(Xplatform[1].+r*cos.(theta),Yplatform[1].+r*sin.(theta),color=[169/255,169/255,169/255])

# savefig("states.svg")

# show()