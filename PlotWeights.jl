# plot Weights 


indexneuroncenter=20; # index of the circle location 

# 	
# 	
# 	                          mm     mm
# 	                          MM     MM
# 	,pP"Ybd  ,p6"bo   ,6"Yb.mmMMmm mmMMmm .gP"Ya `7Mb,od8
# 	8I   `" 6M'  OO  8)   MM  MM     MM  ,M'   Yb  MM' "'
# 	`YMMMa. 8M        ,pm9MM  MM     MM  8M""""""  MM
# 	L.   I8 YM.    , 8M   MM  MM     MM  YM.    ,  MM
# 	M9mmmP'  YMbmd'  `Moo9^Yo.`Mbmo  `Mbmo`Mbmmd'.JMML.
# 	
# 	


weightsvalues=neuronsencodingvector*D;
weightstoindexneurons[j,i]=neuronsencodingvector[indexneuroncenter,:]'*D[:,indexnearestneuron];

# draw a small circle around the iput location to check that activity is centered around input 
radiuscircle=2.;


theta=0:0.001:2*pi;

using PyPlot
using PyCall



fig2 = figure("Line Collection Example")
ax = PyPlot.axes(xlim = (-R-4,R+4),ylim=(-R-4,R+4))

# plot circle 
plot(R*cos.(theta),R*sin.(theta),color="dimgray",zorder=1,lw=2)
plot(neuronscoordinates[:,indexneuroncenter][1].+radiuscircle.*cos.(theta),neuronscoordinates[:,indexneuroncenter][2].+radiuscircle.*sin.(theta),linewidth=2,color="r")

pcolormesh(x,x,weightstoindexneurons)
colorbar()
ax[:set_axis_off]()
savefig("Eigenvectortest$(indexneuroncenter).png")


show()




# 	
# 	                   ,,
# 	                 `7MM
# 	                   MM
# 	 ,p6"bo   ,pW"Wq.  MM  ,pW"Wq.`7Mb,od8 `7MMpMMMb.pMMMb.   ,6"Yb. `7MMpdMAo.
# 	6M'  OO  6W'   `Wb MM 6W'   `Wb MM' "'   MM    MM    MM  8)   MM   MM   `Wb
# 	8M       8M     M8 MM 8M     M8 MM       MM    MM    MM   ,pm9MM   MM    M8
# 	YM.    , YA.   ,A9 MM YA.   ,A9 MM       MM    MM    MM  8M   MM   MM   ,AP
# 	 YMbmd'   `Ybmd9'.JMML.`Ybmd9'.JMML.   .JMML  JMML  JMML.`Moo9^Yo. MMbmmd'
# 	                                                                   MM
# 	                                                                 .JMML.

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
ax = PyPlot.axes(xlim = (-R-4,R+4),ylim=(-R-4,R+4))

# plot circle 
plot(R*cos.(theta),R*sin.(theta),color="dimgray",zorder=1,lw=2)
plot(neuronscoordinates[:,indexneuroncenter][1].+radiuscircle.*cos.(theta),neuronscoordinates[:,indexneuroncenter][2].+radiuscircle.*sin.(theta),linewidth=2,color="r")

pcolormesh(x,x,weightstoindexneurons)
colorbar()
ax[:set_axis_off]()
savefig("Eigenvectortest$(indexneuroncenter).png")


show()



