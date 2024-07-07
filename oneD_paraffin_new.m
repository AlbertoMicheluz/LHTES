clc
close all
clear

%% rect fin
mphopen slice3d-r-a-turb.mph               % Opening the comsol practice

% figure
% mphgeom(model)                       % Plotting the model geometry
% figure
% mphplot(model,'pg3','rangenum',1)    % Plotting the model of the average surface temperature
% table_r=mphtable(model,'tbl1');
expr_r=mphgetexpressions(model.param); % Shows the parameters
% figure
% mphmesh(model)                       % Plotting the model mesh
meshstats_r=mphmeshstats(model);       % Struct that includes the number of elements

avgt=mphtable(model,'tbl1').data(:,2);      %[K]
time_r=mphtable(model, 'tbl1').data(:,1);   %[s]
avgt_o=mphtable(model,'tbl2').data(:,2);    %[K]
avg_r=avgt-273.15;   %[°C]
W_r=mphtable(model,'tbl4').data(:,2);       %[W/m2]
avgT_HTF=mphtable(model,'tbl10').data(:,2);    %[K]
avgT_HTF=avgT_HTF-273.15;   %[°C]

n_fin_r=12;         %number of fins
fin_A_contact_r=65790e-6;   %[m2] contact area single fin
fin_per_r=2582e-3;  %[m] perimeter of the fin's transversal surface
fin_l_r=27.5e-3;    %[m] average fin length along the pipe's radial direction
fin_d_r=18.66e-3;   %[m] consecutive fins distance
V_Cu_com=7.8936e-6; %[m3] 3D model
V_Al_r=8.2638e-5;   %[m3]
V_pcm_r=9.8146e-4;  %[m3]

%% circ fin
mphopen slice3d-c-a-turb.mph               % Opening the comsol practice
 
% figure
% mphgeom(model)                       % Plotting the model geometry
% figure
% mphplot(model,'pg3','rangenum',1)    % Plotting the model of the average surface temperature
% table_c=mphtable(model,'tbl1');
expr_c=mphgetexpressions(model.param); % Shows the parameters
% figure
% mphmesh(model)                       % Plotting the model mesh
meshstats_c=mphmeshstats(model);       % Struct that includes the number of elements

avgtt=mphtable(model,'tbl1').data(:,2);
time_c=mphtable(model, 'tbl1').data(:,1);
avgt_oo=mphtable(model,'tbl2').data(:,2);
avg_c=avgtt-273.15;   %[°C]
W_c=mphtable(model,'tbl4').data(:,2);

n_fin_c=125;        %number of fins
fin_A_contact_c=5701e-6;   %[m2] contact area single fin
fin_per_c=196e-3;   %[m] perimeter of the fin's transversal surface
fin_l_c=21.8e-3;    %[m] average fin length along the pipe's radial direction
fin_d_c=10e-3;      %[m] consecutive fins distance
V_Al_c=6.9000e-5;   %[m3]
V_pcm_c=9.9412e-4;  %[m3]

%%
%Pipe geometry
di = 2*7.45e-3;         %[m] pipe inner diameter
L_tube = 1.29;          %[m] pipe length
A_tube = pi*di^2/4;  %[m2] pipe cross section
n_pipes = 96;           %number of pipes contained in the LHTS used for the discharging phase

%Water properties @65°C
rho = 980.6;            %[kg/m3]
cp = 4184;              %[J/(kg*K)]
k = 0.6455;             %[W/(m*K)]
mu = 433.4e-6;          %[Pa/s] 
alpha = k/(rho*cp);     %[m2/s]

%Operational (initial) values
T_in = 50;             %[°C] HTF inlet temperature
T_initial = 80;        %[°C] Average PCM Initial temperature 
G = 0.03;%0.006;             %[kg/s] HTF mass flow rate in a single pipe
u = G/(rho*A_tube);    %[m/s] HTF velocity in each pipe

%Adimensional numbers
Pr = mu*cp/k;
Re = G/A_tube*di/mu;
Nu_t = 0.023*Re^(4/5)*Pr^(0.4);          %valid for Re>10000

%LHTS features:
%PCM:    
rho_PCM = 858;          %[kg/m3] PCM density
cp_s = 1800;            %[J/(kg*K)] 
cp_l = 1800;%2000;            %[J/(kg*K)]
T_pc = 71;              %[°C] Mean phase change temperature
dT_pc = 3;
T_s = T_pc - dT_pc/2;
T_l = T_pc + dT_pc/2;
Dh_pc = 224e3;            %[J/kg] Latent heat
k_PCM = 0.28;             %[W/mK] PCM thermal conductivity
St=cp_s*dT_pc/Dh_pc;    %Stefan number
%HCM:
rho_Cu = 8978;            %[kg/m3]
rho_Al = 2701;            %[kg/m3]
cp_Cu = 381;              %[J/(kg*K)]
cp_Al = 871;              %[J/(kg*K)]   
%Geometry:
V_pcm_tot = 0.573448;     %[m3]
V_Cu_tot = 4.56648e-3;    %[m3]
V_Al_tot = 0.0443597;     %[m3]
V_pcm = V_pcm_tot/L_tube/n_pipes;         %[m3/m] Vol. of PCM around each meter of pipe
V_Cu = V_Cu_tot/L_tube/n_pipes;           %[m3/m] Vol. of Copper (Cu) around each meter of pipe
V_Al = V_Al_tot/L_tube/n_pipes;           %[m3/m] Vol. of Aluminum (Al) around each meter of pipe

A_exchange=0.14448*6*n_pipes;             %rect fins [m2] Total exchange surface (i.e. overall contact surface between PCM and fins)
% A_exchange=9.4998e-4*6*125*n_pipes;       %circular fins
l_c = V_pcm_tot/A_exchange;               %[m] Characteristic length of the specific LHTS design 
A_pcm_r=6*0.15432;
A_pcm_c=0.787150;
l_r_car=V_pcm_r/A_pcm_r;        %[m3] characteristic length rect fins
l_c_car=V_pcm_c/A_pcm_c;

M_PCM = rho_PCM*V_pcm;               %[kg/m] Mass of PCM around each meter of pipe
M_HCM = rho_Cu*V_Cu + rho_Al*V_Al;   %[kg/m] Mass of high conducting material around each meter of pipe
beta = M_PCM/(M_PCM + M_HCM);

T_ref = 20;                          %[°C] Chosen reference temperature for calculation of LHTS energy content

%Discretization and simulation parameters
t_end = 10800;                                %[s]
dt = 1;                                       %[s]
Nt = (t_end/dt);                              %number of timesteps
t = linspace(0,t_end,Nt);                     %[s] time vector
dz = 0.01;                                       %[m]
n_centroids = (L_tube/dz);                    %number of centroids with finite volume discretization
z=(dz/2:dz:L_tube-dz/2)';    %center of cells distribution
z=[0;z;L_tube];  %consistent vector with finite volume method to include BCs
Nz = size(z);                                 %Number of centroids + 2 boundaries (inlet and outlet)
A_lat = pi*di*dz;                             %[m2] Lateral area of each small cylinder in which the whole pipe is divided

Nul=zeros(Nz(1)-1,1);
for i=1:Nz(1)-1
    Nul(i)=3.66 + (0.0668*di/z(i+1)*Re*Pr)/(1 + 0.04*(di/z(i+1)*Re*Pr)^(2/3));
end
Nu_l = mean(Nul);  %This formula considers the thermal entry length in laminar flow (see pag. 513 Incropera - "Fundamentals of heat and mass transfer")
Nu = (Re<=2300)*Nu_l + (Re>=2300)*Nu_t;   %Nusselt number is evaluated choosing between the laminar and the turbulent value depending on Re number (the "region" between laminar and turbulent is approximated with the turbulent value, which in principle should be applicable only fo Re>10000)
h_conv = k*Nu/di;                         %[W/(m2*K)]

%Initial values
T_old = T_initial*ones(Nz);                   %[°C] Initialization of axial HTF temperature
SOC_0 = ones(Nz(1)-2,1);                           %[-] Initial LHTS state of charge 
SOC = SOC_0;                                     %[-] Initialization of LHTS axial states of charge
E_0 = deltaEtot(T_initial,cp_s,Dh_pc,T_pc,V_Cu,V_Al,V_pcm,rho_PCM)*dz*ones(Nz(1)-2,1);          %[J] Initial energy content in each small volume of PCM-HCM assembly
E_t = E_0;                                       %[J] Initialization of the energy content in each small volume of PCM-HCM assembly

%Construction of problem matrix (A) with BCs
a_low = -u*dt/dz*ones(Nz(1)-1,1);
a_diag = (1 + u*dt/dz)*ones(Nz);
a_diag(1) = 1;
a_diag(end) = 1;     %[-1] indicates last element of a_diag
a_low(end)=-1;     %The last element in the matrix is [-2], because the last element of the array a_low will be cut in the matrix construction 
A = spdiags([[a_low;0], a_diag], [-1,0], Nz(1), Nz(1));    %sparse matrix allows to reduce computational time 
%Right-hand side vector with BCs
b = zeros(Nz);
b(1) = T_in;
b(end) = 0;

%Inizialization of output variables
T_out = T_in*ones(Nt,1);             %[°C] HTF outlet temperature
T_out_k=(T_in+273.15)*ones(Nt,1);   %[K] HTF outlet temperature
T_axial = T_in*ones(Nz(1),Nt);      %[°C] HTF axial temperature
q_axial = zeros(Nz(1)-2,Nt);        %[W/m2] axial heat flux
SOC_axial = ones(Nz(1)-2,Nt);       %[-] axial state of charge of each pipe section
T_wall = T_in*ones(Nz(1)-2,Nt);     %[°C] Wall temperature at the contact between HTF and PCM-HCM assembly
G_t = G*ones(Nt,1);                  %[kg/s] mass flow rate in a single tube in time G(t)
Tin_t = T_in*ones(Nt,1);             %[°C] inlet temperature in time T_in(t)
q_tot = zeros(Nt,1);                 %[W/tube] Total heat transfer rate from each LHTS tube in time 
state_charge = ones(Nt,1);           %[-] Average storage state of charge

DE_ref = (deltaEtot(T_initial,cp_s,Dh_pc,T_pc,V_Cu,V_Al,V_pcm,rho_PCM)-deltaEtot(T_ref,cp_s,Dh_pc,T_pc,V_Cu,V_Al,V_pcm,rho_PCM))*dz*ones(Nz(1)-2,1);   %reference delta energy for linearization of the problem (could also be calculated outside the "for loop")
tau0ref=0.632*(rho_PCM*(l_c*SOC_0).^2*((T_ref<=T_s)*(cp_s*(T_s-T_ref) + cp_l*(T_initial-T_l) + Dh_pc) +((T_ref>T_s) & (T_ref<T_l))*(cp_l*(T_initial-T_l) + Dh_pc*(T_l-T_ref)/dT_pc) +(T_ref>=T_l)*(cp_l*(T_initial-T_ref))))/(k_PCM*(T_initial-T_ref));
tau=tau0ref(1);
Bi=h_conv*l_r_car/k_PCM;        %Biot number rect fins
% Bi=h_conv*l_c_car/k_PCM;      %Biot number circ fins
Fo=k_PCM*tau/(l_r_car^2*rho_PCM*cp_s);      %Fourier number rect fins
% Fo=k_PCM*tau/(l_c_car^2*rho_PCM*cp_s);   %Fourier number circ fins
beta=beta*(0.48*fin_A_contact_r/(l_c*fin_per_r)*(Bi/Fo)^0.2);    %rect fin correction
% beta=beta*(0.48*fin_A_contact_c/(l_c*fin_per_c)*(Bi/Fo)^0.2);    %circ fin correction
tau0ref=tau0ref*(1.3*(St*Fo/Bi)^(1/3));    %tau correction
q_ref=zeros(Nz(1)-2,1);              %reference thermal power value for linearization of the problem (must be calculated inside the "for loop" because it depends on the state of charge)
Tw=zeros(Nz(1)-2,1);
q_wall=zeros(Nz(1)-2,1);        

for j=1:Nt       
for i=1:Nz(1)-2   
    q_ref(i)=(SOC(i)*beta/tau0ref(i)*(log(1/SOC(i) + 1e-10))^((beta-1)/beta))*DE_ref(i);
    Tw(i) = (q_ref(i) - T_ref*q_ref(i)/(T_ref-T_initial) + h_conv*A_lat*T_old(i+1))/(h_conv*A_lat - q_ref(i)/(T_ref-T_initial));  %[°C] Temperature at the contact wall between the PCM and the pipe
    q_wall(i) = h_conv*A_lat*(Tw(i) - T_old(i+1));                 %[W] Thermal power exchanged at the contact wall between the PCM and the pipe
    b(i+1) = T_old(i+1) + dt/(rho*cp*A_tube*dz)*q_wall(i);     %update of right-hand side vector inner elements
end
    T_z = A\b;                      %[°C] calculation of axial HTF temperature T(z)

    %Update variables
    T_old = T_z;
%     T_in = 50                           %[°C] here the value of the HTF inlet temperature can be updated at each time-step if desired
    E_t = E_t - q_wall*dt;               %[J] new energy content of each small volume around a pipe
    E_end = deltaEtot(T_in,cp_s,Dh_pc,T_pc,V_Cu,V_Al,V_pcm,rho_PCM)*dz;               %[J] update of the "final" LHTS energy content (i.e. the ideal LHTS energy content at the end of the discharging process) depending on the new value of the HTF inlet temperature
    SOC = (E_t - E_end)/(E_0(1) - E_end);   %Update of the state of charge of each small volume around the pipe
%     G = 0.006                           %[kg/s] here the value of the HTF mass flow rate in a single tube can be updated at each time-step if desired
%     u = G/(rho*A_tube)                  %[m/s] update of the HTF velocity
    
    %Adimensional numbers and h_conv update
%     Re = G/A_tube*di/mu
%     Nu_t = 0.023*Re**(4/5)*Pr**(0.4)           
%     Nu_l = np.mean(3.66 + (0.0668*di/z[1:-1]*Re*Pr)/(1 + 0.04*(di/z[1:-1]*Re*Pr)**(2/3)))  
%     Nu = (Re<=2300)*Nu_l + (Re>=2300)*Nu_t
%     h_conv = k*Nu/di

    %Update problem matrix and right-end side vector
%     b[0] = T_in
%     a_low = -u*dt/dz*np.ones(Nz)
%     a_diag = (1 + u*dt/dz)*np.ones(Nz)
%     a_diag[0] = 1
%     a_diag[-1] = 1     %[-1] indicates last element of a_diag
%     a_low[-2] = -1     %The last element in the matrix is [-2], because the last element of the array a_low will be cut in the matrix construction 
%     A = sp.sparse.spdiags([a_low, a_diag], [-1,0], Nz, Nz).tocsc()    %sparse matrix allows to reduce computational time 

    
    %Output variables
    T_out(j) = T_z(end);     
    T_out_k(j)=T_out(j)+273.15;
    T_axial(:,j) = T_z;
    q_axial(:,j) = q_wall/(pi*di*dz);  %[W/m2] axial heat flux exchanged at the contact wall between the pipe and the PCM
    SOC_axial(:,j) = SOC;
    T_wall(:,j) = Tw;
    G_t(j) = G;        
    Tin_t(j) = T_in;
    q_tot(j) = sum(q_wall);              
    state_charge(j) = mean(SOC);

end

T_out(1)=T_initial;
T_out_k(1)=T_out(1)+273.15;

figure
plot(t,T_out)
xlim([0,7200])
ylim([T_in,T_initial])
xlabel('Time [s]')
ylabel('Temperature [°C]')

figure
plot(t,state_charge)
xlim([0,7200])
xlabel('Time [s]')
ylabel('SOC')

figure
plot(t,q_tot*n_pipes/1000)
xlim([0,7200])
xlabel('Time [s]')
ylabel('Heat flux [kW]')

%%

A_lateral=pi*di*L_tube/6;   %[m2]
V_Cu_com=7.928e-6;   %[m3] 3D model
V_Al_r=8.2679e-5;    %[m3]
V_pcm_r=9.8146e-4;   %[m3]
Nt_r=7201/dt;
SOC_r=ones(Nt_r,1);
E_0_r = deltaEtot(T_initial,cp_s,Dh_pc,T_pc,V_Cu_com,V_Al_r,V_pcm_r,rho_PCM);
E_t_r=E_0_r;
E_end_r = deltaEtot(T_in,cp_s,Dh_pc,T_pc,V_Cu_com,V_Al_r,V_pcm_r,rho_PCM);

for ii=1:Nt_r
    E_t_r = E_t_r - W_r(ii)*A_lateral*dt;
    SOC_r(ii)=(E_t_r - E_end_r)/(E_0_r - E_end_r);
end

V_Al_c=6.9000e-5;   %[m3]
V_pcm_c=9.9412e-4;  %[m3]
Nt_c=7201/dt;
SOC_c=ones(Nt_c,1);
E_0_c = deltaEtot(T_initial,cp_s,Dh_pc,T_pc,V_Cu_com,V_Al_c,V_pcm_c,rho_PCM);
E_t_c=E_0_c;
E_end_c = deltaEtot(T_in,cp_s,Dh_pc,T_pc,V_Cu_com,V_Al_c,V_pcm_c,rho_PCM);

for ii=1:Nt_r
    E_t_c = E_t_c - W_c(ii)*A_lateral*dt;
    SOC_c(ii)=(E_t_c - E_end_c)/(E_0_c - E_end_c);
end

%%

figure
plot(t,T_out_k,'k')  %outlet T [K]
hold on
plot(time_r,avgt_o,'r')
xlim([0,7200]);
ylim([T_in+273.15,T_initial+273.15])
title('Outlet T comparison')
legend('1D model','Rect. fin')
xlabel('Time [s]')
ylabel('Temperature [K]')
hold off

figure
plot(t,T_out_k,'k')  %outlet T [K]
hold on
plot(time_c,avgt_oo,'b')
xlim([0,7200]);
ylim([T_in+273.15,T_initial+273.15])
title('Outlet T comparison')
legend('1D model','Circ. fin')
xlabel('Time [s]')
ylabel('Temperature [K]')
hold off

figure
plot(t,state_charge,'k')
hold on
plot(time_r,SOC_r,'r')
xlim([0,7200])
ylim([0,1])
xlabel('Time [s]')
ylabel('SOC')
title('SOC comparison')
legend('1D model','Rect. fin')
hold off

figure
plot(t,state_charge,'k')
hold on
plot(time_c,SOC_c,'b')
xlim([0,7200])
ylim([0,1])
xlabel('Time [s]')
ylabel('SOC')
title('SOC comparison')
legend('1D model','Circ. fin')
hold off

figure
plot(time_r,avg_r,'r')
hold on
plot(time_c,avg_c,'b')
xlim([0,7200])
ylim([50,80])
xlabel('Time [s]')
ylabel('Temperature [°C]')
title('PCM average temperature')
legend('Rect. fin','Circ. fin')
hold off

figure
plot(t,q_tot*n_pipes/1000,'k')
hold on
plot(time_r,W_r*A_lateral*6*n_pipes/1000,'r')
xlim([0,7200])
ylim([0,140])
xlabel('Time [s]')
ylabel('Heat flux [kW]')
legend('1D','Rect. fin')
hold off

figure
plot(t,q_tot*n_pipes/1000,'k')
hold on
plot(time_r,W_c*A_lateral*6*n_pipes/1000,'b')
xlim([0,7200])
ylim([0,140])
xlabel('Time [s]')
ylabel('Heat flux [kW]')
legend('1D','Circ. fin')
hold off

%% errors
% 
% T_out_error=zeros(3601,1);
% state_charge_error=zeros(3601,1);
% 
% for iii=1:3601
% T_out_error(iii)=abs((avgt_o(iii)-T_out_k(iii))/(avgt_o(iii)-273.15)*100);          %rect fins error Outlet temperature
% % T_out_error(iii)=abs((avgt_oo(iii)-T_out_k(iii))/(avgt_oo(iii)-273.15)*100);          %circ fins error Outlet temperature
% 
% state_charge_error(iii)=abs(SOC_r(iii)-state_charge(iii));      %rect fins error SOC
% % state_charge_error(iii)=abs(SOC_c(iii)-state_charge(iii));      %circ fins error SOC
% end
% 
% T_out_avg_e=mean(T_out_error);
% state_charge_avg_e=max(state_charge_error);
% 
% figure
% plot(time_r(1:3601),T_out_error)
% 
% figure
% plot(time_r(1:3601),state_charge_error)
