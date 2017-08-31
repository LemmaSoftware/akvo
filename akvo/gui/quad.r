
# Define 
Xc <- function(E0, df, tt, phi, T2) {
	E0 * -sin(2*pi*df*tt + phi) * exp(-tt/T2)
}

Yc <- function(E0, df, tt, phi, T2) {
	E0 * cos(2*pi*df*tt + phi) * exp(-tt/T2)
}

#QI <- function(E0, df, tt, phi, T2, X, Y) {
#	(X-Xc(E0, df, tt, phi, T2))  + (Y-Yc(E0, df, tt, phi, T2))
#}
