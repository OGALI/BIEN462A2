%% Question 1 a)
clear all, close all
n = -100000:100000;
xn = cos(pi/7*n)+ 1/2*cos(pi/6*n);
% xcorr(xn,xn);
% plot(xn)
% crosscorr(xn,xn)
[r, lags] = xcorr(xn);
stem(lags,r);
xlim([-50,50])
title('Autocorrelation Sequence')
xlabel('Time Lag')
ylabel('Sample Autocorrelation')
grid on

%% Question 1 b)
i = 1;
for L = [10 30 100 150 300 600]
    Fs = 10^4;            % Sampling frequency                    
    T = 1/Fs;             % Sampling period       
    t = (0:L-1);        % Time vector
    
    figure(1)
    X = cos(pi/7*t)+ 1/2*cos(pi/6*t);
    Y = fft(X);
    P2 = abs(Y/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    f = Fs*(0:(L/2))/L;
    subplot(3,2,i)
    plot(f,P1) 
    title('Single-Sided Amplitude Spectrum of X(t) L='+string(L))
    xlabel('f (Hz)')
    ylabel('|P1(f)|')
    
    % Plots the signal itself
    figure(2)
    subplot(3,2,i)
    plot(Fs*t(1:L-1),X(1:L-1));
    title('The Signal itself with L=' + string(L));
    xlabel('t (milliseconds)');
    ylabel('X(t)');
    
    i = i+1;
end


%% Question 1 c)
clear all, close all
Fs = 10^4;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = 1500;             % Length of signal
t = (0:L-1)*T;        % Time vector

S = cos(pi/7*10^4*t)+ 1/2*cos(pi/6*10^4*t);
X = S + randn(size(t));

plot(xcorr(X,X))

[r, lags] = xcorr(X);
stem(lags,r);
xlim([-50,50])
title('Autocorrelation Sequence')
xlabel('Time Lag')
ylabel('Sample Autocorrelation')
grid on


% the autocorrelation seems to be periodic due to the periodicity of the
% sine function, there is an impulse at t=0 due to the white noise which is
% perfectly correlated with itself when the time lag is 0.

%% Question 1 d)
clear all, close all
Fs = 10^4;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = 1500;             % Length of signal
t = (0:L-1)*T;        % Time vector

S = cos(pi/7*10^4*t)+ 1/2*cos(pi/6*10^4*t);
X = S + randn(size(t));

figure()
subplot(1,2,1)
periodogram(X)
subplot(1,2,2)
pwelch(X)

%% Question 3 b)
%question: should include negative numbers, put unitstep when taking
%inverse laplace transform
clear all, close all

%create the white noise with specific variance
variance = 2; 
W = sqrt(variance).*randn(100); %Gaussian white noise W
W = W(:);
%histogram(W)




% power spectrum density of Ozz
figure()
% plot(fft(xcorr(W)))
subplot(2,2,1)
periodogram(W)
title('\Phi_{zz} using Periodogram')
subplot(2,2,2)
pwelch(W)
title('\Phi_{zz} using Whelech')

% power spectrum density of Oyy
subplot(2,2,3)
A = tf([1],[1 2]);
syms s t
a = ilaplace(1/(s+2))
funciton = matlabFunction(a)
output = funciton(W);
periodogram(output)
title('\Phi_{yy} using Periodogram')

subplot(2,2,4)
pwelch(output)
title('\Phi_{yy} using Whelch')


figure()
cpsd(W,output)

% Question 3 c)
figure()
subplot(1,2,1)
crosscorr(W, output)
subplot(1,2,2)
crosscorr(output, output)

mean(output)
var(output)

%% Question 5b)
clear all, close all
A = tf(90, [90/3 1])
subplot(1,2,1)
impulse(A)
subplot(1,2,2)
step(A)

%% Question 5c)
close all, clear all
A = tf(90, [90/3 1])
bode(A)

%% Question 5d)
t = linspace(0,200,10000);
y = 0.2*90*(1-exp(-t/(90/3)));
plot(t,y)
title('Response To the Integrate and Fire Model')
xlabel('Time (s)')
ylabel('Potential (mV)')


%% Question 5e)
clear all, close all

t = realmax;
w = 0.00001:0.00001:0.1;

y = 18.*w.*(30.*exp(-t/30)./(1+900.*w.^2)+(-30.*w.*cos(t.*w)+sin(w.*t))./(w.*(1+900.*w.^2)));
ix = find(y>15);




