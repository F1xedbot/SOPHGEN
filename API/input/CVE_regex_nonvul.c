int regexec(Reprog *prog, const char *sp, Resub *sub, int eflags)
{
    Resub scratch;
    int i;
    if (!sub)
        sub = &scratch;
    sub->nsub = prog->nsub;
    for (i = 0; i < MAXSUB; ++i)
        sub->sub[i].sp = sub->sub[i].ep = NULL;
    return match(prog->start, sp, sp, prog->flags | eflags, sub, 0);
}