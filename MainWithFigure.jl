using LightGraphs
using LinearAlgebra

using PyPlot
using Colors
using PyCall
@pyimport numpy as np
@pyimport matplotlib.patches as patch


sigmarange=[2.5 10 ];
qrange= [4 40];
freeparamrange=[1 2 ]; # option 1 depends on the value of the other eigenvalues, option 2 doesnt 

R= 100; # Radius of the circle in cm
r=5;# Radius of the platform  in cm
# parameters :

gamma=1; # discount factor

numberofneurons=500; # number of neurons 
numberofinputs=30; # number of inputs
# numberofneighbours=8; # Is the probability distribution concerns only the nearest neighbours and is 0 elsewhere 

## Create points equally distant throughout the environment 
numberofstates=1000; # number of points 

## INIT ACTIVITY :

# timestep
dt=0.001; # timestep in seconds
tau=1; # time constant of neuron in second
# define gain factor for the activity :
g=0.5; # testing the sensitivity to the slope of the activation function


epsilon=0.05; #  the timescale on which the network activity fades.
alpha=0.05; # input strength 
delayinput=5; # how many tau do we initiate the activity 
delayend=30; # how long do we run the thing 

### parameter for the motion of the activity profile :

# Potential positions of the platform : 
Xplatform=[0.3,0,-0.3,0,0.5,-0.5,0.5,-0.5].*R; # in cm
Yplatform=[0,0.3,0,-0.3,0.5,0.5,-0.5,-0.5].*R;# in cm

# Potential Starting positions of the rat :
Xstart=[0.95,0,-0.95,0].*R; # East, North, West, South
Ystart=[0,0.95,0,-0.95].*R;

#The algorithm places n points, of which the kth point is put at distance sqrt(k-1/2) from the boundary (index begins with k=1), and with polar angle 2*pi*k/phi^2 where phi is the golden ratio. Exception: the last alpha*sqrt(n) points are placed on the outer boundary of the circle, and the polar radius of other points is scaled to account for that. This computation of the polar radius is done in the function radius.
function  radius(k,n,b) # k index of point on the boundary, n number of total points, b number of boundary points
    if k>n-b
        r = 1;            # put on the boundary
    else
        r = sqrt(k-1/2)/sqrt(n-(b+1)/2);     # computation of radius of the different points 
    end
end


# sunflower seed arrangement :
function sunflower(n, R, alpha)   # n number of centers,
    # alpha is indicating how much one cares about the evenness of boundary , chose 2 to have nice trade off
    # R is the radius of the circle in cm
    r=zeros(1,n);
    theta=zeros(1,n);
    b = round(alpha*sqrt(n));      # number of boundary points
    phi = (sqrt(5)+1)/2;           # golden ratio
    for k=1:n
        r[k] = R*radius(k,n,b); # computation of the radius of each point 
        theta[k] = 2*pi*k/phi^2; # computation of the angle of each point 
        
        #plot(r*cos.(theta), r*sin.(theta), "m");
    end
    # scatter(r.*cos.(theta), r.*sin.(theta));#, marker='o', "m");
    X=r.*cos.(theta); 
    Y=r.*sin.(theta);
    return vcat(X, Y)
end

coordstates=sunflower(numberofstates,R,2); # this gives the coordinates as a numberofneurons*2 array 
# order states:
coordstates=sortslices(coordstates,dims=2,by=x->(x[2],x[1])); # this gives the coordinates as a numberofneurons*2 array , the first states having the lowest y value

#  load functions 
# activation function
function activation(g,x)
    return g*max(0.0,x)
end


function complexactivation(x)
    return log(1+exp(x))
end



for sigma in sigmarange
for q in qrange 
for freeparam in freeparamrange 

# 	
# 	
# 	
# 	
# 	 .gP"Ya `7MMpMMMb.`7M'   `MF'
# 	,M'   Yb  MM    MM  VA   ,V
# 	8M""""""  MM    MM   VA ,V
# 	YM.    ,  MM    MM    VVV
# 	 `Mbmmd'.JMML  JMML.   W
# 	
# 	

# first trial we consider an exponential transformation of the distances :
P=zeros(numberofstates,numberofstates);
for i=1:numberofstates
   for j=1:numberofstates

        P[i,j]=sqrt(sum((coordstates[:,i].-coordstates[:,j]).^2)); # compute the euclidean distance between edges 
   end
        P[i,:]=exp.(-P[i,:].^2/(2*sigma^2))./sum(exp.(-P[i,:].^2/(2*sigma^2))); # normalize it to obtain a probability distribution
end

eigenvalues=eigvals(P);
eigenvaluesbis=sort(eigenvalues,  rev=true);
# order eigenvectors
ordering=[findall(x->eigenvaluesbis[k]==x,eigenvalues)[1] for k in 1:numberofstates]; # 
# check that we got this right :
if !(eigenvaluesbis==eigenvalues[ordering])
    println("something is wrong with the order")
    #break
end

# diagonalise matrix :
eigenvectors=eigvecs(P); # obtain eigenvector
eigentest=eigenvectors[:,ordering]; # ordered by order of growing eigenvalue

# define neurons 
indexneurons=sort(unique([rand(1:numberofstates) for k=1:numberofneurons]));
numberofneurons=length(indexneurons); # update the new number of neurons 

neuronscoordinates=coordstates[:,indexneurons];

indexinputs=sort(unique([rand(1:numberofstates) for k=1:numberofinputs]));
numberofinputs=length(indexinputs);

scalevalue=ones(length(eigenvaluesbis[2:end]))./sqrt.((ones(length(eigenvaluesbis[2:end]))-gamma.*eigenvaluesbis[2:end])); # scaling factor to create the new coordinates, exclude the first one that is 1 to avoid problems with the computation
########### Play on the free parameter                                   .JMML.
if freeparam==1 # if option 1 we use 
	freeparameter=sqrt(1.2*(1-gamma*eigenvaluesbis[2])^(-1)); # testing different ranges of freeparameter
	# freeparameter=scalevalue[1]*1.5;
elseif freeparam==2 # if option 2 we use 
	freeparameter=0.001;
end 

newscalevalue=vcat(freeparameter, scalevalue); # changing the first coefficient as being the free parameter to control activity 
newmatrix=Diagonal(newscalevalue); # creating a diagonal matrix with it to be able to multiply each column (=eigenvector) by the right value 
matrixpoints=eigentest*newmatrix; # obtaining new coordinates for the points 
# Now we need to normalise each column to obtain the neurons coordinates 
factornorm=Diagonal([1/norm(matrixpoints[k,:]) for k in 1:size(matrixpoints,1) ]); # taking the norms of the rows and making a diagonal out of it 
matrixneurons=factornorm*matrixpoints; # new coordinates for neurons, the norm of each vector should be 1 now 
# each row is one neuron in the order, each column are their coordinate along each eigenvectors direction
neuronsencodingvector=matrixneurons[indexneurons,1:q]; # coordinates of the neurons 
# each row is one input in the order, each column are their coordinate along each eigenvectors direction
inputsencodingvectorbis=zeros(size(matrixpoints[indexinputs,1:q],2),size(matrixpoints[indexinputs,1:q],1))
inputsencodingvector=transpose!(inputsencodingvectorbis,matrixpoints[indexinputs,1:q]);
# 	
# 	                           ,,            ,,
# 	                           db          `7MM        mm
# 	                                         MM        MM
# 	`7M'    ,A    `MF'.gP"Ya `7MM  .P"Ybmmm  MMpMMMb.mmMMmm ,pP"Ybd
# 	  VA   ,VAA   ,V ,M'   Yb  MM :MI  I8    MM    MM  MM   8I   `"
# 	   VA ,V  VA ,V  8M""""""  MM  WmmmP"    MM    MM  MM   `YMMMa.
# 	    VVV    VVV   YM.    ,  MM 8M         MM    MM  MM   L.   I8
# 	     W      W     `Mbmmd'.JMML.YMMMMMb .JMML  JMML.`MbmoM9mmmP'
# 	                              6'     dP
# 	                              Ybmmmd'

# to obtain the decoding weights, necessity to input different input randomnly scattered through the environment and then compute the resulting steady state activity 
# define vector of activities of the neurons:
global totalactivities
#totalactivities=activation.(g,neuronsencodingvector*inputsencodingvector); # we are storing the activities to compute pseudo inverse blabla
totalactivities=complexactivation.(neuronsencodingvector*inputsencodingvector); # we are storing the activities to compute pseudo inverse blabla

################################ Compute Moore-penrose pseudo inverse ################################
# The psuedo inverse compute the closest solution of Az=c in the least square sense. Our problem is DA=INPUTS , and we need to find D the weights adapted to decoding the activity
# We then will compute : tAtD=tINPUTS , and then tD=tINPUTS*pinv(transpose(A)) 

totalactivitiesbis=zeros(size(totalactivities,2),size(totalactivities,1));
totalactivitiesbis=transpose!(totalactivitiesbis,totalactivities)
pseudoinv=pinv(totalactivitiesbis,0.01);

D=(pseudoinv*inputsencodingvector')'


# 	
# 	           ,,                                          ,,          ,...       ,,        ,,
# 	`7MM"""YMM db                                        `7MM        .d' ""     `7MM      `7MM
# 	  MM    `7                                             MM        dM`          MM        MM
# 	  MM   d `7MM  .P"Ybmmm      ,6"Yb.  `7MMpMMMb.   ,M""bMM       mMMmm,pW"Wq.  MM   ,M""bMM  .gP"Ya `7Mb,od8
# 	  MM""MM   MM :MI  I8       8)   MM    MM    MM ,AP    MM        MM 6W'   `Wb MM ,AP    MM ,M'   Yb  MM' "'
# 	  MM   Y   MM  WmmmP"        ,pm9MM    MM    MM 8MI    MM        MM 8M     M8 MM 8MI    MM 8M""""""  MM
# 	  MM       MM 8M            8M   MM    MM    MM `Mb    MM        MM YA.   ,A9 MM `Mb    MM YM.    ,  MM
# 	.JMML.   .JMML.YMMMMMb      `Moo9^Yo..JMML  JMML.`Wbmd"MML.    .JMML.`Ybmd9'.JMML.`Wbmd"MML.`Mbmmd'.JMML.
# 	              6'     dP
# 	              Ybmmmd'
mkdir("Run_sigma$(sigma)q$(q)freeparam$(freeparam)")

# 	
# 	           ,,                              ,,
# 	         `7MM           mm                 db
# 	           MM           MM
# 	`7MMpdMAo. MM  ,pW"Wq.mmMMmm      .gP"Ya `7MM  .P"Ybmmm .gP"Ya `7MMpMMMb.
# 	  MM   `Wb MM 6W'   `Wb MM       ,M'   Yb  MM :MI  I8  ,M'   Yb  MM    MM
# 	  MM    M8 MM 8M     M8 MM       8M""""""  MM  WmmmP"  8M""""""  MM    MM
# 	  MM   ,AP MM YA.   ,A9 MM       YM.    ,  MM 8M       YM.    ,  MM    MM
# 	  MMbmmd'.JMML.`Ybmd9'  `Mbmo     `Mbmmd'.JMML.YMMMMMb  `Mbmmd'.JMML  JMML.
# 	  MM                                          6'     dP
# 	.JMML.                                        Ybmmmd'



# 	
# 	           ,,                      ,,
# 	         `7MM           mm         db                                    mm
# 	           MM           MM                                               MM
# 	`7MMpdMAo. MM  ,pW"Wq.mmMMmm     `7MM  `7MMpMMMb. `7MMpdMAo.`7MM  `7MM mmMMmm
# 	  MM   `Wb MM 6W'   `Wb MM         MM    MM    MM   MM   `Wb  MM    MM   MM
# 	  MM    M8 MM 8M     M8 MM         MM    MM    MM   MM    M8  MM    MM   MM
# 	  MM   ,AP MM YA.   ,A9 MM         MM    MM    MM   MM   ,AP  MM    MM   MM
# 	  MMbmmd'.JMML.`Ybmd9'  `Mbmo    .JMML..JMML  JMML. MMbmmd'   `Mbod"YML. `Mbmo
# 	  MM                                                MM
# 	.JMML.                                            .JMML.

indexneuroncenter=20; # index of the circle location 
# establish the grid of points in the pool
# draw a small circle around the iput location to check that activity is centered around input 
radiuscircle=2.;
theta=0:0.001:2*pi;
scalarstate=[neuronsencodingvector[indexneuroncenter,:]'*matrixneurons[i,1:q] for i=1:numberofstates]
fig2 = figure("Line Collection Example")
ax = PyPlot.axes(xlim = (-R-R/100,R+R/100),ylim=(-R-R/100,R+R/100))
# plot circle 
plot(R*cos.(theta),R*sin.(theta),color="dimgray",zorder=1,lw=2)
plot(neuronscoordinates[:,indexneuroncenter][1].+radiuscircle.*cos.(theta),neuronscoordinates[:,indexneuroncenter][2].+radiuscircle.*sin.(theta),linewidth=2,color="r")
scatter(coordstates[1,:].-1/2,coordstates[2,:].-1/2,c=scalarstate./255);#,color="r") # add plot of the input 
colorbar()
ax[:set_axis_off]()
savefig("Run_sigma$(sigma)q$(q)freeparam$(freeparam)/Input$(indexneuroncenter).png")

indexneuroncenter=100; # index of the circle location 
# establish the grid of points in the pool
# draw a small circle around the iput location to check that activity is centered around input 
radiuscircle=2.;
theta=0:0.001:2*pi;
scalarstate=[neuronsencodingvector[indexneuroncenter,:]'*matrixneurons[i,1:q] for i=1:numberofstates]
fig2 = figure("Line Collection Example")
ax = PyPlot.axes(xlim = (-R-R/100,R+R/100),ylim=(-R-R/100,R+R/100))
# plot circle 
plot(R*cos.(theta),R*sin.(theta),color="dimgray",zorder=1,lw=2)
plot(neuronscoordinates[:,indexneuroncenter][1].+radiuscircle.*cos.(theta),neuronscoordinates[:,indexneuroncenter][2].+radiuscircle.*sin.(theta),linewidth=2,color="r")
scatter(coordstates[1,:].-1/2,coordstates[2,:].-1/2,c=scalarstate./255);#,color="r") # add plot of the input 
colorbar()
ax[:set_axis_off]()
savefig("Run_sigma$(sigma)q$(q)freeparam$(freeparam)/Input$(indexneuroncenter).png")




# 	
# 	  ,,                ,,
# 	  db                db   mm
# 	                         MM
# 	`7MM  `7MMpMMMb.  `7MM mmMMmm
# 	  MM    MM    MM    MM   MM
# 	  MM    MM    MM    MM   MM
# 	  MM    MM    MM    MM   MM
# 	.JMML..JMML  JMML..JMML. `Mbmo
# 	
# 	
#indexstart=rand(1:4);
#indexplatform=rand(1:length(Xplatform));
indexstart=4;
indexplatform=5;
# find index of the first input state = the nearest state of the official starting points 
indexinput1=findall((minimum(sum((coordstates-repeat(vcat(Xstart[indexstart], Ystart[indexstart]),1,size(coordstates,2))).^2,dims=1)).==sum((coordstates-repeat(vcat(Xstart[indexstart], Ystart[indexstart]),1,size(coordstates,2))).^2,dims=1))[:]);
# find index of the second input state = the nearest state of the official platform point  
indexinput2=findall((minimum(sum((coordstates-repeat(vcat(Xplatform[indexplatform], Yplatform[indexplatform]),1,size(coordstates,2))).^2,dims=1)).==sum((coordstates-repeat(vcat(Xplatform[indexplatform], Yplatform[indexplatform]),1,size(coordstates,2))).^2,dims=1))[:]);
inputencodingvector1=matrixpoints[indexinput1,1:q];
inputencodingvector2=matrixpoints[indexinput2,1:q];

# define trajectory of the maximum 
global trajectory
trajectory=[];
# input the start location: 
global activity 
activity=zeros(numberofneurons,1)
global t
t=0;
while t<delayinput*tau
	global activity
	#activity=activity.*(1-dt/tau).+dt/tau.*activation.(g,neuronsencodingvector*((1-epsilon).*D*activity[:].+alpha.*inputencodingvector1[:]));
	activity=activity.*(1-dt/tau).+dt/tau.*complexactivation.(neuronsencodingvector*((1-epsilon).*D*activity[:].+alpha.*inputencodingvector1[:]));
	global trajectory
	trajectory=push!(trajectory,findall(maximum(activity).==activity)[1]);
	global t
	t=t+dt;
end # end init of activity with the first input 

# 	
# 	                                                                            ,,
# 	`7MMM.     ,MMF'                                                          `7MM
# 	  MMMb    dPMM                                                              MM
# 	  M YM   ,M MM  ,pW"Wq.`7M'   `MF'.gP"Ya       .P"Ybmmm ,pW"Wq.   ,6"Yb.    MM
# 	  M  Mb  M' MM 6W'   `Wb VA   ,V ,M'   Yb     :MI  I8  6W'   `Wb 8)   MM    MM
# 	  M  YM.P'  MM 8M     M8  VA ,V  8M""""""      WmmmP"  8M     M8  ,pm9MM    MM
# 	  M  `YM'   MM YA.   ,A9   VVV   YM.    ,     8M       YA.   ,A9 8M   MM    MM
# 	.JML. `'  .JMML.`Ybmd9'     W     `Mbmmd'      YMMMMMb  `Ybmd9'  `Moo9^Yo..JMML.
# 	                                              6'     dP
# 	                                              Ybmmmd'












end 
end 
end