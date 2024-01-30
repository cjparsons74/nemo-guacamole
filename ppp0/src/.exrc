if &cp | set nocp | endif
let s:cpo_save=&cpo
set cpo&vim
map! <D-v> *
xmap gx <Plug>NetrwBrowseXVis
nmap gx <Plug>NetrwBrowseX
xnoremap <silent> <Plug>NetrwBrowseXVis :call netrw#BrowseXVis()
nnoremap <silent> <Plug>NetrwBrowseX :call netrw#BrowseX(netrw#GX(),netrw#CheckIfRemote(netrw#GX()))
vmap <BS> "-d
vmap <D-x> "*d
vmap <D-c> "*y
vmap <D-v> "-d"*P
nmap <D-v> "*P
let &cpo=s:cpo_save
unlet s:cpo_save
set background=dark
set backspace=2
set expandtab
set fileencodings=ucs-bom,utf-8,default,latin1
set helplang=en
set history=1000
set hlsearch
set ignorecase
set incsearch
set laststatus=2
set modelines=0
set runtimepath=~/.vim,~/.vim/pack/git-plugins/start/test,~/.vim/pack/git-plugins/start/python3,~/.vim/pack/git-plugins/start/plugin,~/.vim/pack/git-plugins/start/ftplugin,~/.vim/pack/git-plugins/start/doc,~/.vim/pack/git-plugins/start/autoload,/usr/share/vim/vimfiles,/usr/share/vim/vim90,/usr/share/vim/vimfiles/after,~/.vim/after
set scrolloff=10
set shiftwidth=4
set showcmd
set showmatch
set smartcase
set statusline=\ %F\ %M\ %Y\ %R%=
set tabstop=4
set wildignore=*.docx,*.jpg,*.png,*.gif,*.pdf,*.pyc,*.exe,*.flv,*.img,*.xlsx
set wildmenu
set wildmode=list:longest
set window=5765414564569022464
" vim: set ft=vim :
