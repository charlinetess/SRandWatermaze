

fig2, axe= PyPlot.subplots(1,length(indexvectors)+1,figsize=(51,10),gridspec_kw= Dict("width_ratios"=>[1, 1, 1, 1, 1, 0.00000001])) #vcat(ones(1,length(indexvectors)),0.05)))
# let image 
global image 

# eigenstate=zeros(length(indexvectors),numberofstates)

# [eigentest[i,indexvector] for i=1:numberofstates for indexvector in indexvectors ]
eigenstate=reshape([eigentest[i,indexvector] for i=1:numberofstates for indexvector in indexvectors ],length(indexvectors),numberofstates)
global cmax, cmin
cmax=maximum(eigenstate)/255;
cmin=minimum(eigenstate)/255;
# norm = plt.Normalize(cmin, cmax)
# reshape([i*j for i=1:2 for j=1:3],3,2)


for indexvector=1:length(indexvectors)

			ax=subplot(1,length(indexvectors)+1,indexvector)

			ax[:set_axis_off]()
			# plot circle 
			ax.plot(R*cos.(theta),R*sin.(theta),ls="-",color="k")#[169/255,169/255,169/255])
			ax.set_xlim(-R-R/100,R+R/100)
			ax.set_ylim(-R-R/100,R+R/100)
			global image 
			image=ax.scatter(coordstates[1,:].-1/2,coordstates[2,:].-1/2,c=eigenstate[indexvector,:]./255,s=5);#,color="r") # this works 
			image=PyPlot.scatter(coordstates[1,:].-1/2,coordstates[2,:].-1/2,c=eigenstate[indexvector,:]./255,s=5);#,color="r") 

			PyPlot.clim(cmin,cmax)
			# quadmesh.set_clim(cmin,cmax)
			#ax.grid(False) 
			# Hide axes ticks
			ax.set_xticks([])
			ax.set_yticks([])
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			#ax.set_title(L"$ \varphi_$(indexvector) $")
			title=latexstring("\$\\varphi_{$(indexvector-1)}\$")
			ax.set_title(title,fontweight="bold", size=20)
end

ax=subplot(1,length(indexvectors)+1,length(indexvectors)+1)
ax[:set_axis_off]()
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
# fig2.subplots_adjust(wspace=0.01)
PyPlot.colorbar(image,ax=ax) # this works but the colorbar is very far away 

show()

