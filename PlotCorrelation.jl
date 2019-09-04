# plot correlation heatmap for a chosen neuron 

indexneuron=500;

correlationmatrix=neuronsencodingvector*transpose(neuronsencodingvector); # matrix of correlations

# draw a small circle around the iput location to check that activity is centered around input 
rad=2.;
theta=0:0.001:2*pi;
cooreltoplot=zeros(l,L); 

counting=0;


for (i,j) in zip(convert.(Int64, neuronscoordinates)[:,1],convert.(Int64, neuronscoordinates)[:,2])


counting=counting+1;
println(counting);
println(i,j);
println(intersect(find(x->x==j,convert.(Int64, neuronscoordinates)[:,2]),find(x->x==i,convert.(Int64, neuronscoordinates)[:,1])));


cooreltoplot[i,j]=correlationmatrix[counting,indexneuron]; # select the indexneurons-th line and make it bigger to see clearer the differences 

end


###### ERROR IN THIS
using PyPlot
using PyCall
@pyimport matplotlib.patches as patch

fig2 = figure("Line Collection Example")
ax2 = axes(xlim = (0,l),ylim=(0,L))

# Create a Rectangle patch
#rect = patch.Rectangle((0,0),l,L,linewidth=1,edgecolor="r",facecolor="none")
#
# Add the patch to the Axes
#ax2[:add_patch](rect)
ax2[:add_artist](rect)
# ax2[:set_xlim]([0,l])
# ax2[:set_ylim]([0,L])
col = Vector{Int}[[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]] # Colors
# Assemble everything into a LineCollection
line_segments2=matplotlib[:collections][:LineCollection](walls,colors=col)

ax2[:add_collection](line_segments2)
#axis("image")
#heatmap(1:1:l,1:1:L,act,aspect_ratio=1)

#plot(coordstates[indexpath,1],coordstates[indexpath,2],"g")
pcolormesh(transpose(cooreltoplot))
colorbar()
#scatter(inputscoordinates[indexinput,:][1],inputscoordinates[indexinput,:][2],color="r") # add plot of the input center 
plot(neuronscoordinates[indexneuron,:][1]+rad.*cos.(theta),neuronscoordinates[indexneuron,:][2]+rad.*sin.(theta),linewidth=0.5,color="r")
savefig("CorrelationPlot$(indexneuron)$(freeparameter)$(numberofneurons).png")
show()
