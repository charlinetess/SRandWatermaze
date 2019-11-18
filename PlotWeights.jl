# plot Weights 


indexneuroncenter=1; # index of the circle location 
# establish the grid of points in the pool
steps=1;
x=[-R+(steps)*(k-1) for k=1:(2*R/steps+1)];
y=zeros(1,length(x));
transpose!(y,x);

# Create the weight matrix :
weightstoindexneurons=zeros(length(x),length(x));

weightsvalues=neuronsencodingvector*D;

for i = 1:length(x)
    for j = 1:length(x)
        # make sure the point is in the pool
        if sqrt((x[i]^2+y[j]^2)) < R

			indexnearestneuron=findall(sum((neuronscoordinates.-transpose(repeat([x[i] y[j]],size(neuronscoordinates,2)))).^2,dims=1)[:].==minimum(sum((neuronscoordinates.-transpose(repeat([x[i] y[j]],size(neuronscoordinates,2)))).^2,dims=1)[:]))[1];

				#weightstoindexneurons[j,i]=weightsvalues[intersect(findall(x->x==j,convert.(Int64, neuronscoordinates)[:,2]),findall(x->x==i,convert.(Int64, neuronscoordinates)[:,1]))[1],indexneuroncenter];
				# then we compute e(corresponding neuron)*D(post synaptic neuron, or here indexneuroncenter)
				
			weightstoindexneurons[j,i]=neuronsencodingvector[indexneuroncenter,:]'*D[:,indexnearestneuron];
        else
            weightstoindexneurons[j,i] = NaN;
    	end
	end
end 


# draw a small circle around the iput location to check that activity is centered around input 
radiuscircle=2.;


theta=0:0.001:2*pi;

using PyPlot
using PyCall



fig2 = figure("Line Collection Example")
ax = PyPlot.axes(xlim = (-R-2,R+2),ylim=(-R-2,R+2))

# plot circle 
plot(R*cos.(theta),R*sin.(theta),color="dimgray",zorder=1,lw=2)
plot(neuronscoordinates[:,indexneuroncenter][1].+radiuscircle.*cos.(theta),neuronscoordinates[:,indexneuroncenter][2].+radiuscircle.*sin.(theta),linewidth=2,color="r")

pcolormesh(x,x,weightstoindexneurons)
colorbar()
ax[:set_axis_off]()
#savefig("Eigenvector$(indexvector).png")
show()

#This works fine
# Test plotting with meshgrid 
# test=[1 2 1 ; 1 1 1 ; 1 1 1];
# using PyPlot
# using PyCall
# pcolormesh(test)
# colorbar()
# show()


