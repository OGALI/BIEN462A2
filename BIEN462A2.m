%% Question 1 a)
clear all, close all
n = 0:50;
xn = cos(pi/7*10^3*n)+ 1/2*cos(pi/6*10^3*n);
xcorr(xn,xn);
plot(xn)



%% Question 1 b)
clear all, close all
% n = 0:100;
% xn = cos(pi/7*10^3*n)+ 1/2*cos(pi/6*10^3*n);
% 
% periodagram(xn)

Fs = 10;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = 10;             % Length of signal
t = (0:L-1)*T;        % Time vector
X = cos(pi/7*10^4*t)+ 1/2*cos(pi/6*10^4*t);
% plot(1000*t(1:50),X(1:50))
% title('Signal Corrupted with Zero-Mean Random Noise')
% xlabel('t (milliseconds)')
% ylabel('X(t)')
Y = fft(X)
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;

subplot(2,2,1)
plot(f,P1) 
title('Single-Sided Amplitude Spectrum of X(t) L=10')
xlabel('f (Hz)')
ylabel('|P1(f)|')



L = 11;             % Length of signal
t = (0:L-1)*T;        % Time vector
X = cos(pi/7*10^3*t)+ 1/2*cos(pi/6*10^3*t);
Y = fft(X)
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;
subplot(2,2,2)
plot(f,P1) 
title('Single-Sided Amplitude Spectrum of X(t) L=11')
xlabel('f (Hz)')
ylabel('|P1(f)|')

L = 100;             % Length of signal
t = (0:L-1)*T;        % Time vector
X = cos(pi/7*10^3*t)+ 1/2*cos(pi/6*10^3*t);
Y = fft(X)
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;
subplot(2,2,3)
plot(f,P1) 
title('Single-Sided Amplitude Spectrum of X(t) L=100')
xlabel('f (Hz)')
ylabel('|P1(f)|')


L = 500;             % Length of signal
t = (0:L-1)*T;        % Time vector
X = cos(pi/7*10^3*t)+ 1/2*cos(pi/6*10^3*t);
Y = fft(X)
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;
subplot(2,2,4)
plot(f,P1) 
title('Single-Sided Amplitude Spectrum of X(t) L=500')
xlabel('f (Hz)')
ylabel('|P1(f)|')


%% Question 1 c)
clear all, close all
Fs = 10;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = 1500;             % Length of signal
t = (0:L-1)*T;        % Time vector

S = 0.7*sin(2*pi*50*t) + sin(2*pi*120*t);
X = S + randn(size(t));

plot(xcorr(X,X))

% the autocorrelation seems to be periodic due to the periodicity of the
% sine function, there is an impulse at t=0 due to the white noise which is
% perfectly correlated with itself when the time lag is 0.

%% Question 1 d)
clear all, close all
Fs = 10;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = 1500;             % Length of signal
t = (0:L-1)*T;        % Time vector

S = 0.7*sin(2*pi*50*t) + sin(2*pi*120*t);
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



% add heaviside unitstep function if needed
% syms t
% a = ilaplace(1/(s^2+2))*heaviside(t)


% power spectrum density of Ozz
figure()
% plot(fft(xcorr(W)))
subplot(2,2,1)
periodogram(W)
title('Ozz using Periodogram')
subplot(2,2,2)
pwelch(W)
title('Ozz using Whelech')

% power spectrum density of Oyy
subplot(2,2,3)
A = tf([1],[1 2]);
syms s t
a = ilaplace(1/(s+2))
funciton = matlabFunction(a)
output = funciton(W);
periodogram(output)
title('Oyy using Periodogram')

subplot(2,2,4)
pwelch(output)
title('Oyy using Whelch')


%crosscorr of unitstep and output
figure()
periodogram(crosscorr(W, output));


% Question 3 c)
figure()
subplot(1,2,1)
crosscorr(W, output)
subplot(1,2,2)
crosscorr(output, output)

