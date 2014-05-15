%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Installation script for matlab GPU k-means clustering Tooolbox
% version 1.0
%
% AUTHOR: 
% Nikolaos Sismanis
% 
% Aristotle University of Thessaloniki
% Faculty of Engineering
% Department of Electical and Computer Engineering
%
% DATE: 
% Jan 2010
%
% CONTACT INFO:
% e-mail: nik_sism@hotmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all

p = path;

add2path= pwd;
add2path = [add2path '/bin'];

path(p, add2path);

savepath

clear all
close all
