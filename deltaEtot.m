function DE=deltaEtot(T,cp_s,Dh_pc,T_pc,V_Cu,V_Al,V_pcm,rho_PCM)

%HCM:
rho_Cu = 8978;            %[kg/m3]
rho_Al = 2701;            %[kg/m3]
cp_Cu = 381;              %[J/(kg*K)]
cp_Al = 871;              %[J/(kg*K)] 

dT_pc=3;
T_ref=20;   %[°C]

T_d=T_ref:0.1:T;

cp_as=cp_s+(Dh_pc/(dT_pc*pi^0.5))*exp(-(T_d-T_pc).^2/dT_pc^2);
DE=rho_Cu*V_Cu*cp_Cu*(T-T_ref)+rho_Al*V_Al*cp_Al*(T-T_ref)+rho_PCM*V_pcm*trapz(T_d,cp_as);