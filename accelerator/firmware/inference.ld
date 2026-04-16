SECTIONS
{
    . = 0x2000;
    .text : {
        * (.start);
        * (.text);
    }
    . = ALIGN(16);
    .data : {
        * (.data);
        * (.rodata);
        * (.rodata.*);
        * (.sdata);
    }
    . = ALIGN(16);
    .bss : {
        * (.bss);
        * (.sbss);
    }
    _end = .;
}