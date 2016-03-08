float rand(vec2 p){
	p+=.2127+p.x+.3713*p.y;
	vec2 r=4.789*sin(789.123*(p));
	return fract(r.x*r.y);
}

float sn(vec2 p){
	vec2 i=floor(p-.5);
	vec2 f=fract(p-.5);
	f = f*f*f*(f*(f*6.0-15.0)+10.0);
	float rt=mix(rand(i),rand(i+vec2(1.,0.)),f.x);
	float rb=mix(rand(i+vec2(0.,1.)),rand(i+vec2(1.,1.)),f.x);
	return mix(rt,rb,f.y);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 uv = fragCoord.xy / iResolution.y;

	uv+=iMouse.xy/ iResolution.xy;
	
	vec2 p=uv.xy*vec2(3.,4.3);
	float f =
	.5*sn(p)
	+.25*sn(2.*p)
	+.125*sn(4.*p)
	+.0625*sn(8.*p)
	+.03125*sn(16.*p)+
	.015*sn(32.*p)
	;
	
	float newT = iGlobalTime*0.4 + sn(vec2(iGlobalTime*1.))*0.1;
	p.x-=iGlobalTime*0.2;
	
	p.y*=1.3;
	float f2=
	.5*sn(p)
	+.25*sn(2.04*p+newT*1.1)
	-.125*sn(4.03*p-iGlobalTime*0.3)
	+.0625*sn(8.02*p-iGlobalTime*0.4)
	+.03125*sn(16.01*p+iGlobalTime*0.5)+
	.018*sn(24.02*p);
	
	float f3=
	.5*sn(p)
	+.25*sn(2.04*p+newT*1.1)
	-.125*sn(4.03*p-iGlobalTime*0.3)
	+.0625*sn(8.02*p-iGlobalTime*0.5)
	+.03125*sn(16.01*p+iGlobalTime*0.6)+
	.019*sn(18.02*p);
	
	float f4 = f2*smoothstep(0.0,1.,uv.y);
	
	vec3 clouds = mix(vec3(-0.4,-0.4,-0.15),vec3(1.4,1.4,1.3),f4*f);
	float lightning = sn((f3)+vec2(pow(sn(vec2(iGlobalTime*4.5)),6.)));

	lightning *= smoothstep(0.0,1.,uv.y+0.5);

	lightning = smoothstep(0.76,1.,lightning);
	lightning=lightning*2.;
	
	vec2 moonp = vec2(0.5,0.8);
	float moon = smoothstep(0.95,0.956,1.-length(uv-moonp));
	vec2 moonp2 = moonp + vec2(0.015, 0);
	moon -= smoothstep(0.93,0.956,1.-length(uv-moonp2));
	moon = clamp(moon, 0., 1.);
	moon += 0.3*smoothstep(0.80,0.956,1.-length(uv-moonp));

	clouds+= pow(1.-length(uv-moonp),1.7)*0.3;

	clouds*=0.8;
	clouds += lightning + moon +0.2;

	float ground = smoothstep(0.07,0.075,f*(p.y-0.98)+0.01);
	
	vec2 newUV = uv;
	newUV.x-=iGlobalTime*0.3;
	newUV.y+=iGlobalTime*3.;
	float strength = sin(iGlobalTime*0.5+sn(newUV))*0.1+0.15;
	
	float rain = sn( vec2(newUV.x*20.1, newUV.y*40.1+newUV.x*400.1-20.*strength ));
	float rain2 = sn( vec2(newUV.x*45.+iGlobalTime*0.5, newUV.y*30.1+newUV.x*200.1 ));	
	rain = strength-rain;
	rain+=smoothstep(0.2,0.5,f4+lightning+0.1)*rain;
	rain += pow(length(uv-moonp),1.)*0.1;
	rain+=rain2*(sin(strength)-0.4)*2.;
	rain = clamp(rain, 0.,0.5)*0.5;
	
	float tree = 0.;
	vec2 treeUV = uv;

	{
		float atree = abs(atan(treeUV.x*59.-85.5,3.-treeUV.y*25.+5.));
		atree +=rand(treeUV.xy*vec2(0.001,1.))*atree;
		tree += clamp(atree, 0.,1.)*0.33;
	}
	{
		float atree = abs(atan(treeUV.x*65.-65.5,3.-treeUV.y*19.+4.));
		atree +=rand(treeUV.xy*vec2(0.001,1.))*atree;
		tree += clamp(atree, 0.,1.)*0.33;
	}
	{
		float atree = abs(atan(treeUV.x*85.-91.5,3.-treeUV.y*21.+4.));
		atree +=rand(treeUV.xy*vec2(0.001,1.))*atree;
		tree += clamp(atree, 0.,1.)*0.34;
	}
	tree = smoothstep(0.6,1.,tree);
	
	
	vec3 painting = tree*ground*(clouds + rain)+clamp(rain*(strength-0.1),0.,1.);
	
	float r=1.-length(max(abs(fragCoord.xy / iResolution.xy*2.-1.)-.5,0.)); 
	painting*=r;
	
	fragColor = BlendChannels(SampleChannel(iChannel0), vec4(painting, 1.));
}
