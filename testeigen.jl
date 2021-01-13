
# fig2,axe = PyPlot.subplots(1,length(indexvectors),figsize=(100,10))#,gridspec_kw={"width_ratios":[1,1, 0.05]})
fig2, axe= PyPlot.subplots(1,length(indexvectors)+1,figsize=(55,10),gridspec_kw= Dict("width_ratios"=>[1, 1, 1, 1, 1, 0.1])) #vcat(ones(1,length(indexvectors)),0.05)))


	#,figsize=(101,10),gridspec_kw= Dict("width_ratios"=>[1, 1, 1, 1, 1, 0.00001])) #vcat(ones(1,length(indexvectors)),0.05)))


# fig2.subplots_adjust(wspace=0.01)
# define scope image 
# let image 
global image 

# eigenstate=zeros(length(indexvectors),numberofstates)

# [eigentest[i,indexvector] for i=1:numberofstates for indexvector in indexvectors ]
eigenstate=reshape([eigentest[i,indexvector] for i=1:numberofstates for indexvector in indexvectors ],length(indexvectors),numberofstates)
cmax=maximum(eigenstate)/255;
cmin=minimum(eigenstate)/255;
norm = plt.Normalize(cmin, cmax)
# reshape([i*j for i=1:2 for j=1:3],3,2)


for indexvector=1:length(indexvectors)

			# ax=axe[indexvector] #subplot(1,length(indexvectors)+1,indexvector)

			axe[indexvector][:set_axis_off]()
			# plot circle 
			axe[indexvector].plot(R*cos.(theta),R*sin.(theta),ls="-",color="k")#[169/255,169/255,169/255])
			axe[indexvector].set_xlim(-R-R/100,R+R/100)
			axe[indexvector].set_ylim(-R-R/100,R+R/100)
			image= axe[indexvector].scatter(coordstates[1,:].-1/2,coordstates[2,:].-1/2,c=eigenstate[indexvector,:]./255,s=5);#,color="r") 
			clim(cmin,cmax)
			#ax.grid(False) 
			# Hide axes ticks
			axe[indexvector].set_xticks([])
			axe[indexvector].set_yticks([])
			axe[indexvector].set_xticklabels([])
			axe[indexvector].set_yticklabels([])
			#ax.set_title(L"$ \varphi_$(indexvector) $")
			title=latexstring("\$\\varphi_{$(indexvector-1)}\$")
			axe[indexvector].set_title(title,fontweight="bold", size=20)
			println(indexvector)
		# colorbar()
end

# fig2.subplots_adjust(right=0.9)
# cbar_ax = fig2.add_axes([0.85, 0, .1, 1])# [left, bottom, width, height] of the new axes. All quantities are in fractions of figure width and height. 
axe[length(indexvectors)+1][:set_axis_off]()
# fig2.colorbar(image, ax=axes.ravel().tolist())
# fig2[:colorbar](ax=cbar_ax,image)
cbar=fig2[:colorbar](image,ax=axe[length(indexvectors)+1])#,cbar_ax)
cbar.ax.tick_params(labelsize=16) 

show()

