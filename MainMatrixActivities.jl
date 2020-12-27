using LightGraphs
using LinearAlgebra


#
#
#     `7MM"""Mq.                                                     mm
#       MM   `MM.                                                    MM
#       MM   ,M9 ,6"Yb.  `7Mb,od8 ,6"Yb.  `7MMpMMMb.pMMMb.  .gP"Ya mmMMmm .gP"Ya `7Mb,od8 ,pP"Ybd
#       MMmmdM9 8)   MM    MM' "'8)   MM    MM    MM    MM ,M'   Yb  MM  ,M'   Yb  MM' "' 8I   `"
#       MM       ,pm9MM    MM     ,pm9MM    MM    MM    MM 8M""""""  MM  8M""""""  MM     `YMMMa.
#       MM      8M   MM    MM    8M   MM    MM    MM    MM YM.    ,  MM  YM.    ,  MM     L.   I8
#     .JMML.    `Moo9^Yo..JMML.  `Moo9^Yo..JMML  JMML  JMML.`Mbmmd'  `Mbmo`Mbmmd'.JMML.   M9mmmP'
#
#
#
#
# $(q)_$(k)_$(g)_$(alpha)_$(freeparameter)_$(epsilon)_$(sigma)

R= 100; # Radius of the circle in cm
r=5;# Radius of the platform  in cm
# parameters :
sigma=2.5; # widths of the bump to compute the probability transition, in case th eprobability transition is a exonential transform of the distances between points 
gamma=1; # discount factor


numberofneurons=500; # number of neurons 

numberofinputs=30; # number of inputs

numberofneighbours=8; # Is the probability distribution concerns only the nearest neighbours and is 0 elsewhere 

## Create points equally distant throughout the environment 
numberofstates=1000; # number of points 

# freeparameter=13;

## INIT ACTIVITY :

# timestep
dt=0.001; # timestep in seconds
tau=1; # time constant of neuron in second
# define gain factor for the activity :
g=0.5; # testing the sensitivity to the slope of the activation function


q=6; # Chose the dimension of the space 


epsilon=0.05; #  the timescale on which the network activity fades.
alpha=0.05; # input strength 



delayinput=10; # how many tau do we initiate the activity 

delayend=30; # how long do we run the thing 

### parameter for the motion of the activity profile :

# Potential positions of the platform : 
Xplatform=[0.3,0,-0.3,0,0.5,-0.5,0.5,-0.5].*R; # in cm
Yplatform=[0,0.3,0,-0.3,0.5,0.5,-0.5,-0.5].*R;# in cm

# Potential Starting positions of the rat :
Xstart=[0.95,0,-0.95,0].*R; # East, North, West, South
Ystart=[0,0.95,0,-0.95].*R;


#
#                                                      ,,
#     `7MM"""YMM                                mm     db
#       MM    `7                                MM
#       MM   d `7MM  `7MM  `7MMpMMMb.  ,p6"bo mmMMmm `7MM  ,pW"Wq.`7MMpMMMb.  ,pP"Ybd
#       MM""MM   MM    MM    MM    MM 6M'  OO   MM     MM 6W'   `Wb MM    MM  8I   `"
#       MM   Y   MM    MM    MM    MM 8M        MM     MM 8M     M8 MM    MM  `YMMMa.
#       MM       MM    MM    MM    MM YM.    ,  MM     MM YA.   ,A9 MM    MM  L.   I8
#     .JMML.     `Mbod"YML..JMML  JMML.YMbmd'   `Mbmo.JMML.`Ybmd9'.JMML  JMML.M9mmmP'
#
#

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

#
#
#      .M"""bgd mm            mm
#     ,MI    "Y MM            MM
#     `MMb.   mmMMmm  ,6"Yb.mmMMmm .gP"Ya  ,pP"Ybd
#       `YMMNq. MM   8)   MM  MM  ,M'   Yb 8I   `"
#     .     `MM MM    ,pm9MM  MM  8M"""""" `YMMMa.
#     Mb     dM MM   8M   MM  MM  YM.    , L.   I8
#     P"Ybmmd"  `Mbmo`Moo9^Yo.`Mbmo`Mbmmd' M9mmmP'
#
#
coordstates=sunflower(numberofstates,R,2); # this gives the coordinates as a numberofneurons*2 array 


# other try : randomn disposition 
#arguments= rand(1,numberofstates)*2*pi;
#radii= sqrt.(rand(1,numberofstates))*R;
#coordstates= [cos.(arguments).*radii; sin.(arguments).*radii]; 

# order states:
coordstates=sortslices(coordstates,dims=2,by=x->(x[2],x[1])); # this gives the coordinates as a numberofneurons*2 array , the first states having the lowest y value


# We actually don't need a graph
#graph= Graph(numberofstates)
#W=zeros(numberofstates,numberofstates);
## create the graph 
#for i=1:numberofstates
#	for j=1:numberofstates
#        # now we loop alogn the number of states
#        add_edge!(graph, i, j)
#        W[i,j]=sqrt(sum((coordstates[:,i].-coordstates[:,j]).^2)); # compute the euclidean distance between edges 
#	end
#end


# we may plot the graph :  # apparently does not work on julia 1.0
#using GraphPlot
#gplot(graph)
#draw("mygraph.png", gplot(graph))

#
#                                                     ,,          ,,                                                                  ,,
#     MMP""MM""YMM                                    db   mm     db                         `7MMM.     ,MMF'         mm              db
#     P'   MM   `7                                         MM                                  MMMb    dPMM           MM
#          MM  `7Mb,od8 ,6"Yb.  `7MMpMMMb.  ,pP"Ybd `7MM mmMMmm `7MM  ,pW"Wq.`7MMpMMMb.        M YM   ,M MM   ,6"Yb.mmMMmm `7Mb,od8 `7MM  `7M'   `MF'
#          MM    MM' "'8)   MM    MM    MM  8I   `"   MM   MM     MM 6W'   `Wb MM    MM        M  Mb  M' MM  8)   MM  MM     MM' "'   MM    `VA ,V'
#          MM    MM     ,pm9MM    MM    MM  `YMMMa.   MM   MM     MM 8M     M8 MM    MM        M  YM.P'  MM   ,pm9MM  MM     MM       MM      XMX
#          MM    MM    8M   MM    MM    MM  L.   I8   MM   MM     MM YA.   ,A9 MM    MM        M  `YM'   MM  8M   MM  MM     MM       MM    ,V' VA.
#        .JMML..JMML.  `Moo9^Yo..JMML  JMML.M9mmmP' .JMML. `Mbmo.JMML.`Ybmd9'.JMML  JMML.    .JML. `'  .JMML.`Moo9^Yo.`Mbmo.JMML.   .JMML..AM.   .MA.
#
#
#################################################################################
########################### Build transition matrix ###############################
#################################################################################

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


# activation function
function activation(g,x)
    return g*max(0.0,x)
end


function complexactivation(x)
    return log(1+exp(x))
end


scalevalue=ones(length(eigenvaluesbis[2:end]))./sqrt.((ones(length(eigenvaluesbis[2:end]))-gamma.*eigenvaluesbis[2:end])); # scaling factor to create the new coordinates, exclude the first one that is 1 to avoid problems with the computation

########### Play on the free parameter 
#freeparameter=sqrt(1.2*(1-gamma*eigenvaluesbis[2])^(-1)); # testing different ranges of freeparameter
freeparameter=scalevalue[1]*1.5;

freeparameter=scalevalue[1];
# freeparameter=0.001;



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

############### initialise activities ##########
# to obtain the decoding weights, necessity to input different input randomnly scattered through the environment and then compute the resulting steady state activity 
# define vector of activities of the neurons:
global totalactivities
#totalactivities=activation.(g,neuronsencodingvector*inputsencodingvector); # we are storing the activities to compute pseudo inverse blabla

totalactivities=complexactivation.(neuronsencodingvector*inputsencodingvector); # we are storing the activities to compute pseudo inverse blabla




#
#                                          ,,                 ,,
#     `7MM"""Mq.                         `7MM                 db
#       MM   `MM.                          MM
#       MM   ,M9 ,pP"Ybd `7MM  `7MM   ,M""bMM  ,pW"Wq.      `7MM  `7MMpMMMb.`7M'   `MF'.gP"Ya `7Mb,od8 ,pP"Ybd  .gP"Ya
#       MMmmdM9  8I   `"   MM    MM ,AP    MM 6W'   `Wb       MM    MM    MM  VA   ,V ,M'   Yb  MM' "' 8I   `" ,M'   Yb
#       MM       `YMMMa.   MM    MM 8MI    MM 8M     M8       MM    MM    MM   VA ,V  8M""""""  MM     `YMMMa. 8M""""""
#       MM       L.   I8   MM    MM `Mb    MM YA.   ,A9       MM    MM    MM    VVV   YM.    ,  MM     L.   I8 YM.    ,
#     .JMML.     M9mmmP'   `Mbod"YML.`Wbmd"MML.`Ybmd9'      .JMML..JMML  JMML.   W     `Mbmmd'.JMML.   M9mmmP'  `Mbmmd'
#
#



################################ Compute Moore-penrose pseudo inverse ################################
# The psuedo inverse compute the closest solution of Az=c in the least square sense. Our problem is DA=INPUTS , and we need to find D the weights adapted to decoding the activity
# We then will compute : tAtD=tINPUTS , and then tD=tINPUTS*pinv(transpose(A)) 

totalactivitiesbis=zeros(size(totalactivities,2),size(totalactivities,1));
totalactivitiesbis=transpose!(totalactivitiesbis,totalactivities)
pseudoinv=pinv(totalactivitiesbis,0.01);

D=(pseudoinv*inputsencodingvector')'


#
#
#     `7MMF'`7MN.   `7MF'`7MMF'MMP""MM""YMM     `7MMF'`7MN.   `7MF'`7MM"""Mq.`7MMF'   `7MF'MMP""MM""YMM
#       MM    MMN.    M    MM  P'   MM   `7       MM    MMN.    M    MM   `MM. MM       M  P'   MM   `7
#       MM    M YMb   M    MM       MM            MM    M YMb   M    MM   ,M9  MM       M       MM
#       MM    M  `MN. M    MM       MM            MM    M  `MN. M    MMmmdM9   MM       M       MM
#       MM    M   `MM.M    MM       MM            MM    M   `MM.M    MM        MM       M       MM
#       MM    M     YMM    MM       MM            MM    M     YMM    MM        YM.     ,M       MM
#     .JMML..JML.    YM  .JMML.   .JMML.        .JMML..JML.    YM  .JMML.       `bmmmmd"'     .JMML.
#
#
####################################################################################################################################
################################# INPUT ACTIVITY : We input the start position for a certain amount of time  ##########################################
####################################################################################################################################



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
#                                                                                                     ,,
#     `7MMM.     ,MMF'                                mm                                            `7MM
#       MMMb    dPMM                                  MM                                              MM
#       M YM   ,M MM  ,pW"Wq.`7M'   `MF'.gP"Ya      mmMMmm ,pW"Wq.       .P"Ybmmm ,pW"Wq.   ,6"Yb.    MM
#       M  Mb  M' MM 6W'   `Wb VA   ,V ,M'   Yb       MM  6W'   `Wb     :MI  I8  6W'   `Wb 8)   MM    MM
#       M  YM.P'  MM 8M     M8  VA ,V  8M""""""       MM  8M     M8      WmmmP"  8M     M8  ,pm9MM    MM
#       M  `YM'   MM YA.   ,A9   VVV   YM.    ,       MM  YA.   ,A9     8M       YA.   ,A9 8M   MM    MM
#     .JML. `'  .JMML.`Ybmd9'     W     `Mbmmd'       `Mbmo`Ybmd9'       YMMMMMb  `Ybmd9'  `Moo9^Yo..JMML.
#                                                                       6'     dP
#                                                                       Ybmmmd'
####################################################################################################################################
################################# then moves the activity profile to the new input location ##########################################
####################################################################################################################################
global t
t=0;
global k
k=0;
global activities_all=[];
while t<delayend*tau
    global activity
    #activity=activity.*(1-dt/tau).+dt/tau.*activation.(g,neuronsencodingvector*((1-epsilon).*D*activity[:].+alpha.*inputencodingvector2[:]));


        activity=activity.*(1-dt/tau).+dt/tau.*complexactivation.(neuronsencodingvector*((1-epsilon).*D*activity[:].+alpha.*inputencodingvector2[:]));


    global t
    t=t+dt;
    global trajectory

    trajectory=push!(trajectory,findall(maximum(activity).==activity)[1]);
    if isinteger(k/100) # every 5 timesteps we store the picture of the activity rofile 
        # establish the grid of points in the pool
        steps=1;
        x=[-R+(steps)*(k-1) for k=1:(2*R/steps+1)];
        # Create matrix activity :
        global act
        act=zeros(length(x),length(x)); 
        y=zeros(1,length(x));
        transpose!(y,x);                    
       for i = 1:length(x)
            for j = 1:length(x)
                # make sure the point is in the pool
                if sqrt((x[i]^2+y[j]^2)) < R
                        indexnearestneuron=findall(sum((neuronscoordinates.-transpose(repeat([x[i] y[j]],size(neuronscoordinates,2)))).^2,dims=1)[:].==minimum(sum((neuronscoordinates.-transpose(repeat([x[i] y[j]],size(neuronscoordinates,2)))).^2,dims=1)[:]))[1];
                        #println(counting);
                        #println(i,j);
                        #  println(intersect(find(x->x==j,convert.(Int64, neuronscoordinates)[:,2]),find(x->x==i,convert.(Int64, neuronscoordinates)[:,1])));
                        global act
                        act[j,i]=activity[indexnearestneuron];
                else
                        global act
                        act[j,i]= NaN;
                end
            end
        end 
        global activities_all
        push!(activities_all,act)

    end

    global k 
    k=k+1;
end # end evolution to the new goal 



