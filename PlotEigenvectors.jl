# chose eigenvectors to pot 
indexvectors=[1,2,3,4,5,6]

theta=0:pi/50:2pi+pi/50;

using PyCall
using PyPlot
using LaTeXStrings
@pyimport matplotlib.patches as patch
@pyimport numpy as np

fig2,axe = PyPlot.subplots(1,length(indexvectors))#,figsize=(25, 5))


# define scope image 
# let image 
global image 

let indexvector=1

#for indexvector in indexvectors	
for ax in axe
	#subplot(1,length(indexvectors),indexvector)

	ax[:set_axis_off]()
	# plot circle 
	ax.plot(R*cos.(theta),R*sin.(theta),ls="--",color=[169/255,169/255,169/255])
	ax.set_xlim(-R,R)
	ax.set_ylim(-R,R)
	#indexvector=4; # index of the vector we want to plot 
	# Create 2-dim aray from the 1D data :
	# define scope of eigenvector 
	steps=1;
	x=[-R+(steps)*(k-1) for k=1:(2*R/steps+1)];
	y=zeros(1,length(x));
	transpose!(y,x);

	# initialise meshgrid 


	# initalize the amplitude map
	# ax.meshgrid(x,y)
	let eigenvector = zeros(length(x),length(x));
		println(indexvector)
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

		indexvector=indexvector+1;
		X,Y = np.meshgrid(x,y)
		global image 
		image=ax.pcolormesh(X,Y,eigenvector,vmin=-0.06,vmax=0.04)
	end# end scope eigenvector  

	#ax.grid(False) 
	# Hide axes ticks
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_xticklabels([])
	ax.set_yticklabels([])

	#ax.set_title(L"$ \varphi_$(indexvector) $")
	title=latexstring("\$\\varphi_{$(indexvector-1)}\$")
		ax.set_title(title,fontweight="bold", size=20)

	# show()

end # end loop through vectorlist

end# end scope indexvector 

global image 
cbar=fig2[:colorbar](ax=axe,image)
cbar.ax.tick_params(labelsize=20) 
show()


savefig("Eigenvector_all.png")

end # end scope image 

#This works fine
# Test plotting with meshgrid 
# test=[1 2 1 ; 1 1 1 ; 1 1 1];
# using PyPlot
# using PyCall
# pcolormesh(test)
# colorbar()
# show()