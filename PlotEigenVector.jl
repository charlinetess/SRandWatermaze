# Plot one eigenvector 
#for 
indexvector=2

#indexvector=9; # index of the vector we want to plot 
clf()
# establish the grid of points in the pool
steps=1;
x=[-R+(steps)*(k-1) for k=1:(2*R/steps+1)];
y=zeros(1,length(x));
transpose!(y,x);
# initalize the amplitude map
eigenvector = zeros(length(x),length(x));

# for each place point in the grid, calculate the critic value
for i = 1:length(x)

    for j = 1:length(x)

        # make sure the point is in the pool
        if sqrt((x[i]^2+y[j]^2)) < R
			# find closest state
			indexneareststate=findall(sum((coordstates.-transpose(repeat([x[i] y[j]],size(coordstates,2)))).^2,dims=1)[:].==minimum(sum((coordstates.-transpose(repeat([x[i] y[j]],size(coordstates,2)))).^2,dims=1)[:]))[1];
			# associate his coordinate on the eigenvector axis :
			eigenvector[j,i]=eigentest[indexneareststate,indexvector];
        else
            eigenvector[j,i] = NaN;
        end
    end
end







theta=0:pi/50:2pi+pi/50;



using PyPlot
using PyCall

fig2 = figure("Line Collection Example")
ax = PyPlot.axes(xlim = (-R,R),ylim=(-R,R))
 gca()[:set_axis_off]()
# plot circle 
plot(R*cos.(theta),R*sin.(theta),ls="--",color=[169/255,169/255,169/255])

pcolormesh(x,x,eigenvector)
colorbar()
savefig("/Users/pmxct2/Documents/FMD-Gerstner/Eigenvector$(indexvector).png")

show()




savefig("Eigenvector$(indexvector).svg")
show()

#This works fine
# Test plotting with meshgrid 
# test=[1 2 1 ; 1 1 1 ; 1 1 1];
# using PyPlot
# using PyCall
# pcolormesh(test)
# colorbar()
# show()



############################################################################
############################################################################
# Try to create video 

# first create matrix : 
steps=1;
x=[-R+(steps)*(k-1) for k=1:(2*R/steps+1)];
y=zeros(1,length(x));
transpose!(y,x);
eigenvectors=[];

for indexvector=1:500
   
    # initalize the amplitude map
    eigenvector = zeros(length(x),length(x));

    # for each place point in the grid, calculate the critic value
    for i = 1:length(x)
    
        for j = 1:length(x)
    
            # make sure the point is in the pool
            if sqrt((x[i]^2+y[j]^2)) < R
                # find closest state
                indexneareststate=findall(sum((coordstates.-transpose(repeat([x[i] y[j]],size(coordstates,2)))).^2,dims=1)[:].==minimum(sum((coordstates.-transpose(repeat([x[i] y[j]],size(coordstates,2)))).^2,dims=1)[:]))[1];
                # associate his coordinate on the eigenvector axis :
                eigenvector[j,i]=eigentest[indexneareststate,indexvector];
            else
                eigenvector[j,i] = NaN;
            end
        end
    end
push!(eigenvectors,eigenvector);
end



using JLD2
using FileIO
save("/Users/pmxct2/Documents/FMD-Gerstner/eigenvectors500.jld2","eigenvectors",eigenvectors)




using JLD2
using FileIO
eigenvector=load("/Users/pmxct2/Documents/FMD-Gerstner/eigenvectors500.jld2")
eigenvectors=eigenvector["eigenvectors"]



##############################################################################
###### THIS WORKS YEAAAH 
####################################################################################

using PyCall
@pyimport matplotlib.animation as anim
using PyPlot

function showanim(filename)
    base64_video = base64encode(open(filename))
    display("text/html", """<video controls src="data:video/x-m4v;base64,$base64_video">""")
end


fig = figure(figsize=(2,2))

function make_frame(i)
    imshow(eigenvectors[:,:,i+1],interpolation="none")
     gca()[:set_axis_off]()

end

withfig(fig) do
    myanim = anim.FuncAnimation(fig, make_frame, frames=42, interval=100)
    myanim[:save]("test3.mp4", bitrate=-1, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
end

showanim("test3.mp4")


