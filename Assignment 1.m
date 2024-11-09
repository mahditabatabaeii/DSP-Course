%% Q1-1

% Parameters
N = 100;                  % Number of points
y = zeros(1, N+1);        % Initialize output y[n] array
x = zeros(1, N+1);        % Initialize input x[n] array
x(1) = 1;                 % Impulse input (x[0] = 1)

% Recursive calculation based on difference equation
for n = 1:N
    if n == 1
        y(n) = x(n);
    elseif n == 2
        y(n) = x(n) + 2*x(n-1) + 0.5*y(n-1);
    elseif n == 3
        y(n) = x(n) + 2*x(n-1) + 0.5*y(n-1) - 0.25*y(n-2);
    else
        y(n) = x(n) + 2*x(n-1) + x(n-3) + 0.5*y(n-1) - 0.25*y(n-2);
    end
end

% Plotting the result
n = 0:N;
stem(n, y, 'filled');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$y[n]$', 'Interpreter', 'latex');
title('System Response $y[n]$ for Given Difference Equation', 'Interpreter', 'latex');
grid on;

%% Q1-2

% Define the numerator and denominator coefficients for H(z)
numerator = [1 2 0 1];      % Coefficients of z^0, z^-1, z^-2, z^-3
denominator = [1 -0.5 0.25]; % Coefficients of z^0, z^-1, z^-2

% Define the transfer function using tf
Hz = tf(numerator, denominator, -1); % -1 indicates Z-transform

% Display the impulse response using impz
N = 100; % Length of impulse response
impulse_response = impz(numerator, denominator, N);

% Plot the impulse response
n = 0:N-1;
stem(n, impulse_response, 'filled');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$h[n]$', 'Interpreter', 'latex');
title('Impulse Response $h[n]$ of the System', 'Interpreter', 'latex');
grid on;

% Find the poles of the transfer function
poles = roots(denominator);

% Display poles and assess stability
disp('Poles of the system:');
disp(poles);

if all(abs(poles) < 1)
    disp('The system is stable because all poles are inside the unit circle.');
else
    disp('The system is unstable because at least one pole is outside the unit circle.');
end


%% Q1-3

% Define the input signal x[n] = (5 + 3*cos(0.2*pi*n) + 4*sin(0.6*pi*n)) * u[n]
N = 100; % Length of the signal
n = 0:N-1; % Time vector
x = (5 + 3*cos(0.2*pi*n) + 4*sin(0.6*pi*n)); % Input signal

% Calculate the output y[n] by applying the system's transfer function to x[n]
y = filter(numerator, denominator, x);

% Plot the input and output signals
figure;
subplot(2,1,1);
stem(n, x, 'filled');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$x[n]$', 'Interpreter', 'latex');
title('Input Signal $x[n]$', 'Interpreter', 'latex');
grid on;

subplot(2,1,2);
stem(n, y, 'filled');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$y[n]$', 'Interpreter', 'latex');
title('System Response $y[n]$ to Input Signal', 'Interpreter', 'latex');
grid on;

%% Q1-4

% Define the frequency range from -pi to pi
num_points = 512; % Number of points for better resolution
[H, w] = freqz(numerator, denominator, num_points, 'whole'); % 'whole' covers -pi to pi

% Shift the frequency and response for plotting from -pi to pi
w = w - pi;
H = fftshift(H); % Shift zero frequency to the center

% Plot the magnitude of the frequency response
figure;
plot(w, abs(H), LineWidth=2);
xlabel('Frequency (radians/sample)');
ylabel('Magnitude');
title('Magnitude of Frequency Response |H(e^{j\omega})|');
grid on;

%% Q1-5

% Define the frequency range from -pi to pi
num_points = 512; % Number of points for better resolution
[H, w] = freqz(numerator, denominator, num_points, 'whole'); % 'whole' covers -pi to pi

% Shift the frequency and response for plotting from -pi to pi
w = w - pi;
H = fftshift(H); % Shift zero frequency to the center

% Calculate the phase response (wrapped and unwrapped)
phase_wrapped = angle(H);           % Wrapped phase
phase_unwrapped = unwrap(angle(H)); % Unwrapped phase

% Plot the wrapped phase response
figure;
subplot(2, 1, 1);
plot(w, phase_wrapped, LineWidth = 2);
xlabel('Frequency (radians/sample)');
ylabel('Phase (radians)');
title('Wrapped Phase Response');
grid on;

% Plot the unwrapped phase response
subplot(2, 1, 2);
plot(w, phase_unwrapped, LineWidth = 2);
xlabel('Frequency (radians/sample)');
ylabel('Phase (radians)');
title('Unwrapped Phase Response');
grid on;

%% Q1-6

% Compute the zeros, poles, and gain using tf2zp
[zeros, poles, gain] = tf2zpk(numerator, denominator);

% Display the zeros, poles, and gain for reference
disp('Zeros of the system:');
disp(zeros);
disp('Poles of the system:');
disp(poles);
disp('Gain of the system:');
disp(gain);

% Plot the zeros and poles using pzplot
figure;
pzplot(tf(numerator, denominator, -1)); % -1 indicates Z-domain
title('Pole-Zero Plot of the System');
grid on;

%% Q2

% Define the transfer function coefficients
numerator = [1 0 -1];        % Coefficients of 1 - z^(-2)
denominator = [1 0.9 0.6 0.05]; % Coefficients of 1 + 0.9z^(-1) + 0.6z^(-2) + 0.05z^(-3)

% Use residuez to get the partial fraction expansion
[residues, poles, direct_term] = residuez(numerator, denominator);

% Display the results
disp('Residues:');
disp(residues);
disp('Poles:');
disp(poles);

% Generate impulse response from partial fractions if necessary
N = 20; % Define the length of the impulse response
impulse_response = impz(numerator, denominator, N);

% Plot the impulse response
n = 0:N-1;
stem(n, impulse_response, 'filled');
xlabel('$n$', 'Interpreter', 'latex');
ylabel('$h[n]$', 'Interpreter', 'latex');
title('Impulse Response $h[n]$ of the System using Partial Fraction Expansion', 'Interpreter', 'latex');
grid on;

%% Q3-1

% Time vector
t = linspace(0, T, 1000);

% Instantaneous frequency
f_t = f1 + mu * t;

% Plot
figure;
plot(t, f_t);
xlabel('Time (s)');
ylabel('Instantaneous Frequency (Hz)');
title('Instantaneous Frequency of the Chirp Signal');

%% Q3-3

% Parameters
f1 = 4e3;         % Initial frequency in Hz (4 kHz)
mu = 600e3;       % Chirp rate in Hz/s (600 kHz/s)
T = 0.05;         % Total time duration in seconds (50 ms)

% Parameters for the signal
phi = 0;          % Assume phase phi is 0 for simplicity
fs = 8e3;         % Sampling frequency in Hz (8 kHz)
t_sampled = 0:1/fs:T; % Sampled time vector

% Sampled signal x(t) using sampled time vector
x_sampled = cos(pi * mu * t_sampled.^2 + 2 * pi * f1 * t_sampled + phi);

% Plot continuous chirp signal
figure;

subplot(2, 1, 1);
plot(t_sampled, x_sampled); % Continuous chirp
ylim([-1.2 1.2])
xlabel('Time (s)');
ylabel('Amplitude');
title('Continuous Chirp Signal');
grid on;

% Plot sampled chirp signal
subplot(2, 1, 2);
stem(t_sampled, x_sampled, 'filled'); % Sampled chirp
ylim([-1.2 1.2])
xlabel('Time (s)');
ylabel('Amplitude');
title('Sampled Chirp Signal');
grid on;

%% Q3-5

% Calculate apparent instantaneous frequency modulo sampling rate
f_alias = mod(f_t, fs);

% Plot apparent frequency due to sampling
figure;
plot(t, f_alias);
xlabel('Time (s)');
ylabel('Apparent Frequency (Hz)');
title('Apparent Frequency of the Sampled Signal (Aliasing Check)');


%% Q4-1

% Define the range of n
n = -50:50; % You can adjust the range as needed

% Define the signals x1[n] and x2[n]
x1 = (sin(pi * n / 10) .^ 2) ./ ((pi * n / 10) .^ 2);
x2 = sin(pi * n / 10) ./ (pi * n / 10);

% Handle n = 0 separately to avoid division by zero
x1(n == 0) = 1; % Using limit as n -> 0 for x1[n]
x2(n == 0) = 1; % Using limit as n -> 0 for x2[n]

% Calculate Fourier Transforms using FFT
N = 1024; % Number of points for FFT
X1_fft = fftshift(fft(x1, N));
X2_fft = fftshift(fft(x2, N));

% Frequency range in [-pi, pi]
omega = linspace(-pi, pi, N);

% Plot the signals x1[n] and x2[n]
figure;
subplot(2, 1, 1);
stem(n, x1, 'filled');
xlabel('n');
ylabel('x_1[n]');
title('Signal x_1[n]');

subplot(2, 1, 2);
plot(omega, abs(X1_fft));
xlabel('\omega');
ylabel('|X_1(e^{j\omega})|');
title('Magnitude of Fourier Transform of x_1[n]');

% Plot the magnitude of Fourier Transform of x1[n] and x2[n]
figure;

subplot(2, 1, 1);
stem(n, x2, 'filled');
xlabel('n');
ylabel('x_2[n]');
title('Signal x_2[n]');

subplot(2, 1, 2);
plot(omega, abs(X2_fft), LineWidth=1.5);
xlabel('\omega');
ylabel('|X_2(e^{j\omega})|');
title('Magnitude of Fourier Transform of x_2[n]');

%% Q4-2

x2(n == 0) = 1; % Handle n = 0 separately to avoid division by zero

% Define y1[n] = x2[2n]
% To calculate y1 correctly, we need to redefine the range for n to account for the downsampling
n_y1 = -25:25; % Adjusted range for y1, since we are sampling every 2nd n in x2
y1 = sin(pi * (2 * n_y1) / 10) ./ (pi * (2 * n_y1) / 10);
y1(n_y1 == 0) = 1; % Handle n_y1 = 0 separately to avoid division by zero

% Define y2[n] based on even/odd conditions
y2 = zeros(size(n));
for i = 1:length(n)
    if mod(n(i), 2) == 0
        y2(i) = x2(n(i) / 2 + 51); % Index adjustment for zero-centered n
    end
end

% Define y3[n] = x2[n] * sin(2π * 0.3 * n)
y3 = x2 .* sin(2 * pi * 0.3 * n);

% Plot the signals x2[n], y1[n], y2[n], and y3[n]
figure;
subplot(4, 1, 1);
stem(n, x2, 'filled');
xlabel('n');
ylabel('x_2[n]');
title('Signal x_2[n]');

subplot(4, 1, 2);
stem(n_y1, y1, 'filled');
xlabel('n');
ylabel('y_1[n]');
title('Signal y_1[n] = x_2[2n]');

subplot(4, 1, 3);
stem(n, y2, 'filled');
xlabel('n');
ylabel('y_2[n]');
title('Signal y_2[n]');

subplot(4, 1, 4);
stem(n, y3, 'filled');
xlabel('n');
ylabel('y_3[n]');
title('Signal y_3[n] = x_2[n] * sin(2\pi \cdot 0.3 \cdot n)');

% Fourier Transform calculations
N = 1024; % Number of points for FFT
omega = linspace(-pi, pi, N);

% Fourier Transforms of the signals
X2_fft = fftshift(fft(x2, N));
Y1_fft = fftshift(fft(y1, N));
Y2_fft = fftshift(fft(y2, N));
Y3_fft = fftshift(fft(y3, N));

% Plot Fourier Transforms
figure;
subplot(4, 1, 1);
plot(omega, abs(X2_fft), LineWidth=1.5);
xlabel('\omega');
ylabel('|X_2(e^{j\omega})|');
title('Magnitude of Fourier Transform of x_2[n]');

subplot(4, 1, 2);
plot(omega, abs(Y1_fft), LineWidth=1.5);
xlabel('\omega');
ylabel('|Y_1(e^{j\omega})|');
title('Magnitude of Fourier Transform of y_1[n]');

subplot(4, 1, 3);
plot(omega, abs(Y2_fft), LineWidth=1.5);
xlabel('\omega');
ylabel('|Y_2(e^{j\omega})|');
title('Magnitude of Fourier Transform of y_2[n]');

subplot(4, 1, 4);
plot(omega, abs(Y3_fft), LineWidth=1.5);
xlabel('\omega');
ylabel('|Y_3(e^{j\omega})|');
title('Magnitude of Fourier Transform of y_3[n]');

%% Q5-2 , Q5-4

% Define parameters
f1 = 500;
f2 = 1000;
Fs = 1000;              % Sampling frequency (5 kHz)
Ts = 1 / Fs;            % Sampling period (0.0002 seconds)

% Define continuous and sampled time vectors
t_continuous = 0:1e-5:0.01;  % Continuous time vector with very fine resolution
t_samples = 0:Ts:0.01;       % Sampled time vector based on Fs

% Generate the continuous signal
x_continuous = sin(2 * pi * f1 * t_continuous) + sin(2 * pi * f2 * t_continuous);

% Generate the sampled signal
x_samples = sin(1000 * pi * t_samples) + sin(2000 * pi * t_samples);

% Set the interpolation time vector to match the continuous time vector
t_interp = t_continuous;

% Perform sinc interpolation using the provided sinc_interpolation function
x_interp1 = sinc_interpolation(x_samples, t_samples, t_interp, Ts);
limited_x_interp1 = limited_sinc_interpolation(x_samples, t_samples, t_interp, Ts, 9);

% Plotting
figure;

% Plot original continuous signal
subplot(4, 1, 1);
plot(t_continuous, x_continuous, 'LineWidth', 1.5);
xlabel('Time (s)', 'Interpreter', 'latex');
ylabel('Amplitude', 'Interpreter', 'latex');
title('Original Continuous Signal $x(t)$', 'Interpreter', 'latex');
grid on;

% Plot sampled signal
subplot(4, 1, 2);
stem(t_samples, x_samples, 'filled');
xlabel('Time (s)', 'Interpreter', 'latex');
ylabel('Amplitude', 'Interpreter', 'latex');
title('Sampled Points $x[n]$', 'Interpreter', 'latex');
grid on;

% Plot interpolated signal
subplot(4, 1, 3);
plot(t_interp, x_interp1, 'LineWidth', 1.2);
xlabel('Time (s)', 'Interpreter', 'latex');
ylabel('Amplitude', 'Interpreter', 'latex');
title('Interpolated Signal using Sinc Interpolation $\hat{x}(t)$', 'Interpreter', 'latex');
grid on;

% Plot interpolated signal
subplot(4, 1, 4);
plot(t_interp, limited_x_interp1, 'LineWidth', 1.2);
xlabel('Time (s)', 'Interpreter', 'latex');
ylabel('Amplitude', 'Interpreter', 'latex');
title('Interpolated Signal using Limited Sinc Interpolation $\hat{x}(t)$', 'Interpreter', 'latex');
grid on;


%% Functions

% Q5-1

function y_interp = sinc_interpolation(x_samples, t_samples, t_interp, Ts)
    % Initialize the interpolated signal vector
    y_interp = zeros(size(t_interp));
    
    % Perform sinc interpolation for each point in t_interp
    for k = 1:length(t_interp)
        % Calculate the sinc kernel for each sample point
        sinc_kernel = sinc((t_interp(k) - t_samples) / Ts);
        
        % Calculate the interpolated value at time t_interp(k)
        y_interp(k) = sum(x_samples .* sinc_kernel);
    end
end

% Q5-3

function x_interp = limited_sinc_interpolation(x_samples, t_samples, t_interp, Ts, L)
    % Initialize the interpolated signal vector
    x_interp = zeros(size(t_interp));
    
    % Perform limited sinc interpolation for each point in t_interp
    for i = 1:length(t_interp)
        % Find the nearest sample index in t_samples to the current t_interp(i)
        [~, nearest_idx] = min(abs(t_samples - t_interp(i)));
        
        % Define the range of indices for the limited sinc kernel
        start_idx = max(nearest_idx - L, 1);
        end_idx = min(nearest_idx + L, length(t_samples));
        
        % Calculate the sinc kernel for the limited range
        sinc_kernel = sinc((t_interp(i) - t_samples(start_idx:end_idx)) / Ts);
        
        % Calculate the interpolated value at time t_interp(i) using limited kernel
        x_interp(i) = sum(x_samples(start_idx:end_idx) .* sinc_kernel);
    end
end

